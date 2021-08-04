#include <common.hpp>
#include <data_reader.hpp>
#include <data_readers/async_reader/async_reader.hpp>
#include <data_readers/async_reader/async_reader_adapter.hpp>
#include <data_readers/async_reader/async_reader_common.hpp>
#include <data_readers/async_reader/split_label_dense_sparse.hpp>
#include <resource_manager.hpp>
#include <tensor2.hpp>
#include <utils.hpp>

namespace {
using namespace HugeCTR;

class RawPtrWrapper : public HugeCTR::TensorBuffer2 {
 public:
  RawPtrWrapper(void* ptr) : ptr_(ptr) {}
  bool allocated() const override { return true; }
  void* get_ptr() override { return ptr_; }

 private:
  void* ptr_;
};

class RawPtrBuffer : public HugeCTR::TensorBuffer2 {
 public:
  RawPtrBuffer(size_t size_bytes) { CK_CUDA_THROW_(cudaMalloc(&ptr_, size_bytes)); }
  bool allocated() const override { return true; }
  void* get_ptr() override { return ptr_; }
  ~RawPtrBuffer() override { cudaFree(ptr_); }

 private:
  void* ptr_;
};

class DerivedPtrBuffer : public HugeCTR::TensorBuffer2 {
 public:
  DerivedPtrBuffer(void* ptr, const std::shared_ptr<RawPtrBuffer>& buffer)
      : ptr_(ptr), buffer_(buffer) {}
  bool allocated() const override { return true; }
  void* get_ptr() override { return ptr_; }

 private:
  void* ptr_;
  std::shared_ptr<RawPtrBuffer> buffer_;
};
}  // namespace

namespace HugeCTR {

template <typename SparseType>
AsyncReader<SparseType>::AsyncReader(std::string fname, size_t batch_size, size_t label_dim,
                                     size_t dense_dim, std::vector<DataReaderSparseParam>& params,
                                     bool mixed_precision,
                                     const std::shared_ptr<ResourceManager>& resource_manager,
                                     int num_threads, int num_batches_per_thread,
                                     size_t io_block_size, int io_depth, int io_alignment,
                                     bool shuffle, bool wait_for_gpu_idle, Alignment_t aligned)
    : resource_manager_(resource_manager),
      mixed_precision_(mixed_precision),
      batch_size_(batch_size),
      completion_events_(resource_manager->get_local_gpu_count()),
      schedule_events_(resource_manager->get_local_gpu_count()) {
  assert(batch_size_ % resource_manager_->get_global_gpu_count() == 0);
  assert(params.size() == 1);
  static_assert(sizeof(LabelType) == sizeof(InputType));

  batch_size_per_dev_ = batch_size_ / resource_manager_->get_global_gpu_count();

  size_t dense_dim_align8 = dense_dim;
  if (aligned == Alignment_t::Auto) dense_dim_align8 = (dense_dim + 7) / 8 * 8;
  size_t sparse_dim = params[0].slot_num;
  sample_size_items_ =
      label_dim + dense_dim + sparse_dim * (sizeof(SparseType) / sizeof(InputType));
  size_t batch_size_bytes = sample_size_items_ * sizeof(InputType) * batch_size;

  reader_impl_ = std::make_unique<AsyncReaderImpl>(
      fname, batch_size_bytes, resource_manager.get(), num_threads, num_batches_per_thread,
      io_block_size, io_depth, io_alignment, shuffle, wait_for_gpu_idle);

  for (uint32_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    auto local_gpu = resource_manager_->get_local_gpu(i);
    CudaDeviceContext ctx(local_gpu->get_device_id());
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&completion_events_[i], cudaEventDisableTiming));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&schedule_events_[i], cudaEventDisableTiming));

    auto my_dev_id = resource_manager_->get_gpu_global_id_from_local_id(i);

    auto label_buffer_ = std::make_shared<RawPtrBuffer>(batch_size * label_dim * sizeof(LabelType));
    label_tensors_.emplace_back(
        Tensor2<LabelType>({batch_size, label_dim}, label_buffer_).shrink());

    auto label_buffer_offset = my_dev_id * (batch_size_per_dev_ * label_dim);
    auto label_buffer_per_dev = std::make_shared<DerivedPtrBuffer>(
        ((LabelType*)(label_buffer_->get_ptr()) + label_buffer_offset), label_buffer_);
    label_tensors_per_dev_.emplace_back(
        Tensor2<LabelType>({batch_size_per_dev_, label_dim}, label_buffer_per_dev).shrink());

    if (mixed_precision_) {
      auto dense_buffer_ =
          std::make_shared<RawPtrBuffer>(batch_size * dense_dim_align8 * sizeof(__half));
      dense_tensors_.emplace_back(
          Tensor2<__half>({batch_size, dense_dim_align8}, dense_buffer_).shrink());

      auto dense_buffer_offset = my_dev_id * (batch_size_per_dev_ * dense_dim_align8);
      auto dense_buffer_per_dev = std::make_shared<DerivedPtrBuffer>(
          ((__half*)dense_buffer_->get_ptr() + dense_buffer_offset), dense_buffer_);
      dense_tensors_per_dev_.emplace_back(
          Tensor2<__half>({batch_size_per_dev_, dense_dim_align8}, dense_buffer_per_dev).shrink());

    } else {
      auto dense_buffer_ =
          std::make_shared<RawPtrBuffer>(batch_size * dense_dim_align8 * sizeof(float));
      dense_tensors_.emplace_back(
          Tensor2<float>({batch_size, dense_dim_align8}, dense_buffer_).shrink());

      auto dense_buffer_offset = my_dev_id * (batch_size_per_dev_ * dense_dim_align8);
      auto dense_buffer_per_dev = std::make_shared<DerivedPtrBuffer>(
          ((float*)dense_buffer_->get_ptr() + dense_buffer_offset), dense_buffer_);
      dense_tensors_per_dev_.emplace_back(
          Tensor2<float>({batch_size_per_dev_, dense_dim_align8}, dense_buffer_per_dev).shrink());
    }

    auto sparse_buffer_ =
        std::make_shared<RawPtrBuffer>(batch_size * sparse_dim * sizeof(SparseType));

    auto value_tensor = Tensor2<SparseType>({batch_size, sparse_dim}, sparse_buffer_);
    auto dummy_row_offset_tensor = Tensor2<SparseType>();
    std::shared_ptr<size_t> dummy_nnz(new size_t);
    sparse_tensors_.emplace_back(
        SparseTensor<SparseType>(value_tensor, dummy_row_offset_tensor, dummy_nnz));
  }

  // zero-initialization
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    const auto local_gpu = resource_manager_->get_local_gpu(i);
    if (mixed_precision_) {
      Tensor2<__half> tensor = Tensor2<__half>::stretch_from(dense_tensors_[i]);
      CK_CUDA_THROW_(cudaMemsetAsync(tensor.get_ptr(), 0,
                                     tensor.get_num_elements() * sizeof(__half),
                                     local_gpu->get_memcpy_stream()));
    } else {
      Tensor2<float> tensor = Tensor2<float>::stretch_from(dense_tensors_[i]);
      CK_CUDA_THROW_(cudaMemsetAsync(tensor.get_ptr(), 0, tensor.get_num_elements() * sizeof(float),
                                     local_gpu->get_memcpy_stream()));
    }
  }
}

template <typename SparseType>
long long AsyncReader<SparseType>::read_a_batch_to_device_delay_release() {
  auto batch = reader_impl_->get_batch();
  if (batch.size_bytes == 0) {
    reader_impl_->reset();
    reader_impl_->load_async();
    batch = reader_impl_->get_batch();
  }

  int num_local_gpus = resource_manager_->get_local_gpu_count();

#pragma omp parallel for num_threads(num_local_gpus)
  for (int i = 0; i < num_local_gpus; i++) {
    auto local_gpu = resource_manager_->get_local_gpu(i);
    CudaCPUDeviceContext ctx(local_gpu->get_device_id());

    current_batch_size_ = batch.size_bytes / (sample_size_items_ * sizeof(InputType));
    auto ptr_wrap =
        std::make_shared<RawPtrWrapper>(reinterpret_cast<InputType*>(batch.dev_data[i]));

    if (mixed_precision_) {
      split_3_way<__half, SparseType>(
          Tensor2<LabelType>::stretch_from(label_tensors_[i]),
          Tensor2<__half>::stretch_from(dense_tensors_[i]), sparse_tensors_[i].get_value_tensor(),
          Tensor2<InputType>({current_batch_size_, sample_size_items_}, ptr_wrap),
          local_gpu->get_stream());
    } else {
      split_3_way<float, SparseType>(
          Tensor2<LabelType>::stretch_from(label_tensors_[i]),
          Tensor2<float>::stretch_from(dense_tensors_[i]), sparse_tensors_[i].get_value_tensor(),
          Tensor2<InputType>({current_batch_size_, sample_size_items_}, ptr_wrap),
          local_gpu->get_stream());
    }

    // CK_CUDA_THROW_(cudaStreamSynchronize(local_gpu->get_stream()));
  }
  // TODO: here we may need a GPU barrier if there's no stream sync
  return current_batch_size_;
}

template <typename SparseType>
long long AsyncReader<SparseType>::get_full_batchsize() const {
  return batch_size_;
}

template <typename SparseType>
void AsyncReader<SparseType>::ready_to_collect() {
  auto raw_device_id = reader_impl_->get_last_batch_device();
  auto local_gpu = resource_manager_->get_local_gpu(raw_device_id);
  CudaDeviceContext ctx(local_gpu->get_device_id());
  CK_CUDA_THROW_(cudaEventRecord(completion_events_[raw_device_id], local_gpu->get_stream()));

  reader_impl_->finalize_batch(&completion_events_[raw_device_id]);
}

template <typename SparseType>
long long AsyncReader<SparseType>::read_a_batch_to_device() {
  auto result = read_a_batch_to_device_delay_release();
  ready_to_collect();
  return result;
}

template <typename SparseType>
void AsyncReader<SparseType>::schedule_here(cudaStream_t stream, int raw_device_id) {
  CK_CUDA_THROW_(cudaEventRecord(schedule_events_[raw_device_id], stream));
  reader_impl_->wait_for_gpu_event(&schedule_events_[raw_device_id], raw_device_id);
}

template <typename SparseType>
void AsyncReader<SparseType>::schedule_here_graph(cudaStream_t stream, int raw_device_id) {
  CK_CUDA_THROW_(
      cudaEventRecordWithFlags(schedule_events_[raw_device_id], stream, cudaEventRecordExternal));
}

template <typename SparseType>
void AsyncReader<SparseType>::update_schedule_graph(int raw_device_id) {
  reader_impl_->wait_for_gpu_event(&schedule_events_[raw_device_id], raw_device_id);
}

template <typename SparseType>
long long AsyncReader<SparseType>::get_current_batchsize_per_device(size_t local_id) {
  long long batchsize_per_device = batch_size_ / resource_manager_->get_global_gpu_count();
  size_t global_id = resource_manager_->get_gpu_global_id_from_local_id(local_id);
  long long remain_samples = current_batch_size_ - global_id * batchsize_per_device;
  if (remain_samples >= batchsize_per_device) {
    return batchsize_per_device;
  } else if (remain_samples > 0) {
    return remain_samples;
  } else {
    return 0;
  }
}

template <typename SparseType>
TensorScalarType AsyncReader<SparseType>::get_scalar_type() const {
  return TensorScalarTypeFunc<SparseType>::get_type();
};
template <typename SparseType>
bool AsyncReader<SparseType>::is_started() const {
  return reader_impl_->is_currently_loading();
}
template <typename SparseType>
void AsyncReader<SparseType>::start() {
  reader_impl_->load_async();
}
template <typename SparseType>
std::vector<TensorBag2> AsyncReader<SparseType>::get_label_tensors() const {
  return label_tensors_per_dev_;
}
template <typename SparseType>
std::vector<TensorBag2> AsyncReader<SparseType>::get_dense_tensors() const {
  return dense_tensors_per_dev_;
}

// template <typename SparseType>
// std::vector<TensorBag2> AsyncReader<SparseType>::get_value_tensors() const {
//   return sparse_tensors_;
// }

template <typename SparseType>
SparseTensors<SparseType> AsyncReader<SparseType>::get_value_tensors() const {
  return sparse_tensors_;
}

template <typename SparseType>
void AsyncReader<SparseType>::create_drwg_norm(std::string file_list, Check_t check_type,
                                               bool start_reading_from_beginning) {}
template <typename SparseType>
void AsyncReader<SparseType>::create_drwg_raw(std::string file_name, long long num_samples,
                                              bool float_label_dense, bool data_shuffle,
                                              bool start_reading_from_beginning) {}
#ifndef DISABLE_CUDF
template <typename SparseType>
void AsyncReader<SparseType>::create_drwg_parquet(std::string file_list,
                                                  const std::vector<long long> slot_offset,
                                                  bool start_reading_from_beginning) {}
#endif
template <typename SparseType>
void AsyncReader<SparseType>::set_source(std::string file_list) {}

template <typename SparseType>
AsyncReader<SparseType>::~AsyncReader() {
  // Underlying reader mush be destroyed BEFORE the events
  reader_impl_.reset(nullptr);
  for (auto& e : completion_events_) {
    cudaEventDestroy(e);
  }
  for (auto& e : schedule_events_) {
    cudaEventDestroy(e);
  }
}

template class AsyncReader<uint32_t>;
template class AsyncReader<long long>;

}  // namespace HugeCTR

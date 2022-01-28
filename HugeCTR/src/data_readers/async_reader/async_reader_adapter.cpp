#include <common.hpp>
#include <data_reader.hpp>
#include <data_readers/async_reader/async_reader.hpp>
#include <data_readers/async_reader/async_reader_adapter.hpp>
#include <data_readers/async_reader/async_reader_common.hpp>
#include <data_readers/async_reader/split_label_dense_sparse.hpp>
#include <resource_manager.hpp>
#include <tensor2.hpp>
#include <utils.hpp>

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
      batch_size_per_dev_(batch_size_ / resource_manager->get_global_gpu_count()),
      completion_events_(resource_manager->get_local_gpu_count()),
      schedule_events_(resource_manager->get_local_gpu_count()),
      split_schedule_events_(resource_manager->get_local_gpu_count()),
      graph_complete_events_(resource_manager->get_local_gpu_count()),
      d2d_schedule_events_(resource_manager->get_local_gpu_count()),
      split_streams_(resource_manager->get_local_gpu_count()),
      graphs_(2 * resource_manager->get_local_gpu_count()) {  // double buffered
  assert(batch_size_ % resource_manager_->get_global_gpu_count() == 0);
  assert(params.size() == 1);
  static_assert(sizeof(LabelType) == sizeof(InputType));

  size_t dense_dim_align8 = dense_dim;
  if (aligned == Alignment_t::Auto) dense_dim_align8 = (dense_dim + 7) / 8 * 8;
  size_t sparse_dim = params[0].slot_num;
  sample_size_items_ =
      label_dim + dense_dim + sparse_dim * (sizeof(SparseType) / sizeof(InputType));
  size_t batch_size_bytes = sample_size_items_ * sizeof(InputType) * batch_size;

  label_dim_ = label_dim;
  dense_dim_ = dense_dim_align8;
  sparse_dim_ = sparse_dim;

  reader_impl_ = std::make_unique<AsyncReaderImpl>(
      fname, batch_size_bytes, resource_manager.get(), num_threads, num_batches_per_thread,
      io_block_size, io_depth, io_alignment, shuffle, wait_for_gpu_idle);

  for (uint32_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    auto local_gpu = resource_manager_->get_local_gpu(i);
    CudaDeviceContext ctx(local_gpu->get_device_id());
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&completion_events_[i], cudaEventDisableTiming));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&schedule_events_[i], cudaEventDisableTiming));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&split_schedule_events_[i], cudaEventDisableTiming));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&graph_complete_events_[i], cudaEventDisableTiming));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&d2d_schedule_events_[i], cudaEventDisableTiming));
    CK_CUDA_THROW_(cudaStreamCreate(&split_streams_[i]));

    auto label_dense_buffer = std::make_shared<RawPtrBuffer>(
        batch_size_per_dev_ *
        (label_dim * sizeof(LabelType) +
         dense_dim_align8 * (mixed_precision ? sizeof(__half) : sizeof(float))));
    auto dense_buffer = std::make_shared<RawPtrWrapper>(
        (LabelType*)(label_dense_buffer->get_ptr()) + batch_size_per_dev_ * label_dim);

    label_tensors_.emplace_back(
        Tensor2<LabelType>({batch_size_per_dev_, label_dim}, label_dense_buffer).shrink());
    if (mixed_precision_) {
      dense_tensors_.emplace_back(
          Tensor2<__half>({batch_size_per_dev_, dense_dim_align8}, dense_buffer).shrink());
    } else {
      dense_tensors_.emplace_back(
          Tensor2<float>({batch_size_per_dev_, dense_dim_align8}, dense_buffer).shrink());
    }

    auto sparse_buffer_ =
        std::make_shared<RawPtrBuffer>(batch_size * sparse_dim * sizeof(SparseType));
    auto value_tensor = Tensor2<SparseType>({batch_size, sparse_dim}, sparse_buffer_);

    auto dummy_row_offset_tensor = Tensor2<SparseType>();
    std::shared_ptr<size_t> dummy_nnz(new size_t);
    sparse_tensors_.emplace_back(
        SparseTensor<SparseType>(value_tensor, dummy_row_offset_tensor, dummy_nnz).shrink());
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
  current_batch_size_ = batch.size_bytes / (sample_size_items_ * sizeof(InputType));

  int num_local_gpus = resource_manager_->get_local_gpu_count();
#pragma omp parallel for num_threads(num_local_gpus)
  for (int i = 0; i < num_local_gpus; i++) {
    auto local_gpu = resource_manager_->get_local_gpu(i);
    CudaCPUDeviceContext ctx(local_gpu->get_device_id());
    auto global_dev_id = resource_manager_->get_gpu_global_id_from_local_id(i);

    if (precomputing_) {
      if (index_processor_->get_queue_size() != get_total_queue_size()) {
        auto& graph = graphs_[queue_id_ * num_local_gpus + i];
        // ensures we schedule this async processing at the correct location
        CK_CUDA_THROW_(cudaStreamWaitEvent(split_streams_[i], split_schedule_events_[i]));
        index_processor_->split3way(batch, queue_id_, i, split_streams_[i]);
        if (!graph.initialized) {
          graph.capture(
              [&, this, i](cudaStream_t graph_stream) {
                // Index calculation
                index_processor_->calculate_indices(batch, queue_id_, i, graph_stream);
                // Copy precomputed tensors for the next iteration, wait until we are safe to
                // overwrite
                CK_CUDA_THROW_(cudaStreamWaitEvent(graph_stream, d2d_schedule_events_[i],
                                                   cudaEventWaitExternal));
                index_processor_->finalize(label_tensors_[i], dense_tensors_[i], sparse_tensors_[i],
                                           queue_id_, i, graph_stream);
                CK_CUDA_THROW_(cudaEventRecordWithFlags(graph_complete_events_[i], graph_stream,
                                                        cudaEventRecordExternal));
              },
              split_streams_[i]);
        }
        graph.exec(split_streams_[i]);
        CK_CUDA_THROW_(cudaStreamWaitEvent(local_gpu->get_stream(), graph_complete_events_[i]));
      } else {
        // The queue size of extra processor is exactly the same as for the DR
        // No graphs here
        if (!batch.cached) {
          index_processor_->split3way(batch, queue_id_, i, local_gpu->get_stream());
          index_processor_->calculate_indices(batch, queue_id_, i, local_gpu->get_stream());
        }
        index_processor_->finalize(label_tensors_[i], dense_tensors_[i], sparse_tensors_[i],
                                   queue_id_, i, local_gpu->get_stream());
      }
    } else {  // synchronous processing
      auto ptr_wrap =
          std::make_shared<RawPtrWrapper>(reinterpret_cast<InputType*>(batch.dev_data[i]));

      if (mixed_precision_) {
        split_3_way<__half, SparseType>(
            Tensor2<LabelType>::stretch_from(label_tensors_[i]),
            Tensor2<__half>::stretch_from(dense_tensors_[i]),
            SparseTensor<SparseType>::stretch_from(sparse_tensors_[i]).get_value_tensor(),
            Tensor2<InputType>({current_batch_size_, sample_size_items_}, ptr_wrap),
            global_dev_id * batch_size_per_dev_, (global_dev_id + 1) * batch_size_per_dev_,
            local_gpu->get_stream());
      } else {
        split_3_way<float, SparseType>(
            Tensor2<LabelType>::stretch_from(label_tensors_[i]),
            Tensor2<float>::stretch_from(dense_tensors_[i]),
            SparseTensor<SparseType>::stretch_from(sparse_tensors_[i]).get_value_tensor(),
            Tensor2<InputType>({current_batch_size_, sample_size_items_}, ptr_wrap),
            global_dev_id * batch_size_per_dev_, (global_dev_id + 1) * batch_size_per_dev_,
            local_gpu->get_stream());
      }
    }
  }

  if (precomputing_) {
    queue_id_ = (queue_id_ + 1) % index_processor_->get_queue_size();
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
  cudaStream_t processing_stream =
      precomputing_ ? split_streams_[raw_device_id] : local_gpu->get_stream();
  CK_CUDA_THROW_(cudaEventRecord(completion_events_[raw_device_id], processing_stream));

  reader_impl_->finalize_batch(&completion_events_[raw_device_id]);
}

template <typename SparseType>
long long AsyncReader<SparseType>::read_a_batch_to_device() {
  auto result = read_a_batch_to_device_delay_release();
  ready_to_collect();
  return result;
}

template <typename SparseType>
void AsyncReader<SparseType>::schedule_precompute_here(cudaStream_t stream, int raw_device_id,
                                                       bool from_graph) {
  unsigned int flags = from_graph ? cudaEventRecordExternal : 0;
  CK_CUDA_THROW_(cudaEventRecordWithFlags(split_schedule_events_[raw_device_id], stream, flags));
}

template <typename SparseType>
void AsyncReader<SparseType>::schedule_d2d_here(cudaStream_t stream, int raw_device_id,
                                                bool from_graph) {
  unsigned int flags = from_graph ? cudaEventRecordExternal : 0;
  CK_CUDA_THROW_(cudaEventRecordWithFlags(d2d_schedule_events_[raw_device_id], stream, flags));
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
void AsyncReader<SparseType>::register_extra_processing(
    const std::shared_ptr<hybrid_embedding::IndexProcessor<SparseType>>& proc) {
  index_processor_ = proc;
  precomputing_ = true;
}

template <typename SparseType>
size_t AsyncReader<SparseType>::get_total_queue_size() {
  return reader_impl_->get_num_buffers();
}

template <typename SparseType>
bool AsyncReader<SparseType>::is_mixed_precision() {
  return mixed_precision_;
}

template <typename SparseType>
void AsyncReader<SparseType>::get_dimensions(size_t& label_dim, size_t& dense_dim,
                                             size_t& sparse_dim, size_t& sample_size_items) {
  label_dim = label_dim_;
  dense_dim = dense_dim_;
  sparse_dim = sparse_dim_;
  sample_size_items = sample_size_items_;
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
  if (!this->is_started()) {
    reader_impl_->load_async();
  }
}
template <typename SparseType>
std::vector<TensorBag2> AsyncReader<SparseType>::get_label_tensors() const {
  return label_tensors_;
}
template <typename SparseType>
std::vector<TensorBag2> AsyncReader<SparseType>::get_dense_tensors() const {
  return dense_tensors_;
}

// template <typename SparseType>
// std::vector<TensorBag2> AsyncReader<SparseType>::get_value_tensors() const {
//   return sparse_tensors_;
// }

template <typename SparseType>
SparseTensors<SparseType> AsyncReader<SparseType>::get_value_tensors() const {
  SparseTensors<SparseType> tmp_tensors;
  tmp_tensors.reserve(sparse_tensors_.size());
  for (auto& st : sparse_tensors_) {
    tmp_tensors.emplace_back(SparseTensor<SparseType>::stretch_from(st));
  }
  return tmp_tensors;
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

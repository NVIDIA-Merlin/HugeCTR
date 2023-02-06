/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
      d2d_schedule_events_(resource_manager->get_local_gpu_count()),
      s3w_streams_(resource_manager->get_local_gpu_count()),
      d2d_streams_(resource_manager->get_local_gpu_count()),
      cache_buffers_(false) {
  assert(batch_size_ % resource_manager_->get_global_gpu_count() == 0);
  assert(params.size() == 1);
  static_assert(sizeof(LabelType) == sizeof(InputType));

  size_t dense_dim_align8 = dense_dim;
  if (aligned == Alignment_t::Auto) dense_dim_align8 = (dense_dim + 7) / 8 * 8;
  size_t sparse_dim = params[0].slot_num;
  sample_size_items_ = label_dim + dense_dim + sparse_dim;
  size_t batch_size_bytes = sample_size_items_ * sizeof(InputType) * batch_size;

  label_dim_ = label_dim;
  dense_dim_ = dense_dim_align8;
  sparse_dim_ = sparse_dim;

  // const std::vector<FileSource>& data_files,
  //                 const std::shared_ptr<ResourceManager>& resource_manager,
  //                 size_t batch_size,
  //                 size_t num_threads_per_file,
  //                 size_t num_batches_per_thread

  reader_impl_ = std::make_unique<AsyncReaderImpl>(
      fname, batch_size_bytes, resource_manager.get(), num_threads, num_batches_per_thread,
      io_block_size, io_depth, io_alignment, shuffle, wait_for_gpu_idle);

  for (uint32_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    auto local_gpu = resource_manager_->get_local_gpu(i);
    CudaDeviceContext ctx(local_gpu->get_device_id());
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&completion_events_[i], cudaEventDisableTiming));
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&schedule_events_[i], cudaEventDisableTiming));
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&split_schedule_events_[i], cudaEventDisableTiming));
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&d2d_schedule_events_[i], cudaEventDisableTiming));

    // set default stream
    s3w_streams_[i] = local_gpu->get_stream();
    d2d_streams_[i] = local_gpu->get_stream();

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
  }

  // zero-initialization
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    const auto local_gpu = resource_manager_->get_local_gpu(i);
    CudaDeviceContext ctx(local_gpu->get_device_id());
    if (mixed_precision_) {
      Tensor2<__half> tensor = Tensor2<__half>::stretch_from(dense_tensors_[i]);
      HCTR_LIB_THROW(cudaMemset(tensor.get_ptr(), 0, tensor.get_num_elements() * sizeof(__half)));
    } else {
      Tensor2<float> tensor = Tensor2<float>::stretch_from(dense_tensors_[i]);
      HCTR_LIB_THROW(cudaMemset(tensor.get_ptr(), 0, tensor.get_num_elements() * sizeof(float)));
    }
  }

  set_tensor_buffering(1);
}

template <typename SparseType>
void AsyncReader<SparseType>::set_tensor_buffering(size_t num_batches_to_buffer) {
  // If the number of buffers exceeds or is equal to number of batches in our dataset, then we
  // may as well cache them so we only execute the 'split_3_way' kernel once.
  cache_buffers_ = num_batches_to_buffer >= reader_impl_->get_num_batches();
  init_batch_tensors(num_batches_to_buffer);
}

template <typename SparseType>
void AsyncReader<SparseType>::init_batch_tensors(size_t num_inflight) {
  inflight_batch_tensors_.resize(num_inflight);

  for (auto& batch_tensors : inflight_batch_tensors_) {
    batch_tensors.tag = SIZE_MAX;  // Invalid

    for (uint32_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
      auto local_gpu = resource_manager_->get_local_gpu(i);
      CudaDeviceContext ctx(local_gpu->get_device_id());

      auto label_dense_buffer = std::make_shared<RawPtrBuffer>(
          batch_size_per_dev_ * (label_dim_ * sizeof(LabelType) +
                                 dense_dim_ * (mixed_precision_ ? sizeof(__half) : sizeof(float))));
      auto dense_buffer = std::make_shared<RawPtrWrapper>(
          (LabelType*)(label_dense_buffer->get_ptr()) + batch_size_per_dev_ * label_dim_);

      batch_tensors.label_tensors.emplace_back(
          Tensor2<LabelType>({batch_size_per_dev_, label_dim_}, label_dense_buffer).shrink());
      if (mixed_precision_) {
        batch_tensors.dense_tensors.emplace_back(
            Tensor2<__half>({batch_size_per_dev_, dense_dim_}, dense_buffer).shrink());
      } else {
        batch_tensors.dense_tensors.emplace_back(
            Tensor2<float>({batch_size_per_dev_, dense_dim_}, dense_buffer).shrink());
      }

      // Zero-initialize in case we're aligning dense tensor
      auto& dense_tensor_bag = batch_tensors.dense_tensors.back();
      HCTR_LIB_THROW(
          cudaMemset(dense_tensor_bag.get_ptr(), 0, dense_tensor_bag.get_size_in_bytes()));

      auto sparse_buffer_ =
          std::make_shared<RawPtrBuffer>(batch_size_ * sparse_dim_ * sizeof(SparseType));
      auto value_tensor = Tensor2<SparseType>({batch_size_, sparse_dim_}, sparse_buffer_);

      auto dummy_row_offset_tensor = Tensor2<SparseType>();
      std::shared_ptr<size_t> dummy_nnz(new size_t);
      batch_tensors.sparse_tensors.emplace_back(
          SparseTensor<SparseType>(value_tensor, dummy_row_offset_tensor, dummy_nnz).shrink());
    }
  }

  // Needed for get_value_tensors() on construction
  current_sparse_tensors_ = inflight_batch_tensors_.at(0).sparse_tensors;
}

template <typename SparseType>
long long AsyncReader<SparseType>::read_a_batch_to_device_delay_release() {
  auto batch = reader_impl_->get_batch();
  if (batch.size_bytes == 0) {
    reader_impl_->reset();
    reader_impl_->load_async();
    batch = reader_impl_->get_batch();
  }

  if (cache_buffers_) {
    // TODO: replace with cache policy like LRU when number of batches exceeds what we can store
    inflight_id_ = batch.id;
  } else {
    inflight_id_ = (inflight_id_ + 1) % inflight_batch_tensors_.size();  // FIFO
  }

  BatchTensors& batch_tensors = inflight_batch_tensors_.at(inflight_id_);

  size_t current_batch_id = static_cast<size_t>(batch.id);
  current_batch_size_ = batch.size_bytes / (sample_size_items_ * sizeof(InputType));
  current_sparse_tensors_ = batch_tensors.sparse_tensors;
  current_batch_cached_ = (current_batch_id == batch_tensors.tag) && cache_buffers_;

  int num_local_gpus = resource_manager_->get_local_gpu_count();
#pragma omp parallel for num_threads(num_local_gpus)
  for (int i = 0; i < num_local_gpus; i++) {
    auto local_gpu = resource_manager_->get_local_gpu(i);
    CudaCPUDeviceContext ctx(local_gpu->get_device_id());
    auto global_dev_id = resource_manager_->get_gpu_global_id_from_local_id(i);

    const cudaStream_t& stream = s3w_streams_[i];

    // schedule at correct place in iteration
    HCTR_LIB_THROW(cudaStreamWaitEvent(stream, split_schedule_events_[i]));

    if (!current_batch_cached_) {  // data can be cached for eval

      auto ptr_wrap =
          std::make_shared<RawPtrWrapper>(reinterpret_cast<InputType*>(batch.dev_data[i]));

      // To save memory we're going to use the space in the Data for the unprocessed
      //  sparse features, and then run to_unique_categories essentially in place
      //    auto current_batch_size = batch.size_bytes / (sample_size_items_ * sizeof(dtype));
      //    auto in_place_tensor = my_data.samples;
      //    in_place_tensor.reset_shape({current_batch_size, sparse_dim_});
      if (mixed_precision_) {
        split_3_way<__half, SparseType>(
            Tensor2<LabelType>::stretch_from(batch_tensors.label_tensors[i]),
            Tensor2<__half>::stretch_from(batch_tensors.dense_tensors[i]),
            SparseTensor<SparseType>::stretch_from(batch_tensors.sparse_tensors[i])
                .get_value_tensor(),
            Tensor2<InputType>({current_batch_size_, sample_size_items_}, ptr_wrap),
            global_dev_id * batch_size_per_dev_, (global_dev_id + 1) * batch_size_per_dev_, stream);
      } else {
        split_3_way<float, SparseType>(
            Tensor2<LabelType>::stretch_from(batch_tensors.label_tensors[i]),
            Tensor2<float>::stretch_from(batch_tensors.dense_tensors[i]),
            SparseTensor<SparseType>::stretch_from(batch_tensors.sparse_tensors[i])
                .get_value_tensor(),
            Tensor2<InputType>({current_batch_size_, sample_size_items_}, ptr_wrap),
            global_dev_id * batch_size_per_dev_, (global_dev_id + 1) * batch_size_per_dev_, stream);
      }
    }

    auto sparse_ready_event = local_gpu->get_event("sparse_tensors_ready");
    HCTR_LIB_THROW(cudaEventRecord(sparse_ready_event, stream));

    auto d2d_stream = d2d_streams_[i];

    // Need result from split-3-way
    HCTR_LIB_THROW(cudaStreamWaitEvent(d2d_stream, sparse_ready_event));

    // we are safe to overwrite
    HCTR_LIB_THROW(cudaStreamWaitEvent(d2d_stream, d2d_schedule_events_[i]));

    // batch.dev_data can be reused
    HCTR_LIB_THROW(cudaEventRecord(completion_events_[i], d2d_stream));

    // isn't part of hybrid embedding
    assign_dense_and_label_tensors(batch_tensors.label_tensors[i], batch_tensors.dense_tensors[i],
                                   i, d2d_stream);

    auto tensors_ready_event = local_gpu->get_event("bottom_MLP_tensors_ready");
    HCTR_LIB_THROW(cudaEventRecord(tensors_ready_event, d2d_stream));
  }

  batch_tensors.tag = current_batch_id;
  return current_batch_size_;
}

template <typename SparseType>
void AsyncReader<SparseType>::set_schedule_streams(cudaStream_t s3w_stream, cudaStream_t d2d_stream,
                                                   int raw_device_id) {
  s3w_streams_[raw_device_id] = s3w_stream;
  d2d_streams_[raw_device_id] = d2d_stream;
}

template <typename SparseType>
void AsyncReader<SparseType>::assign_dense_and_label_tensors(TensorBag2 label_tensor,
                                                             TensorBag2 dense_tensor,
                                                             int raw_device_id,
                                                             cudaStream_t stream) {
  auto& dst_label_tensor = label_tensors_[raw_device_id];
  auto& dst_dense_tensor = dense_tensors_[raw_device_id];

  // TODO: allocate tensors together
  if ((char*)dst_label_tensor.get_ptr() + dst_label_tensor.get_size_in_bytes() ==
      (char*)dst_dense_tensor.get_ptr()) {
    HCTR_LIB_THROW(
        cudaMemcpyAsync(dst_label_tensor.get_ptr(), label_tensor.get_ptr(),
                        dst_label_tensor.get_size_in_bytes() + dense_tensor.get_size_in_bytes(),
                        cudaMemcpyDeviceToDevice, stream));
  } else {
    HCTR_LIB_THROW(cudaMemcpyAsync(dst_label_tensor.get_ptr(), label_tensor.get_ptr(),
                                   dst_label_tensor.get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                   stream));

    HCTR_LIB_THROW(cudaMemcpyAsync(dst_dense_tensor.get_ptr(), dense_tensor.get_ptr(),
                                   dst_dense_tensor.get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                   stream));
  }
}

template <typename SparseType>
long long AsyncReader<SparseType>::get_full_batchsize() const {
  return batch_size_;
}

template <typename SparseType>
void AsyncReader<SparseType>::stream_wait_sparse_tensors(cudaStream_t stream, int raw_device_id,
                                                         bool from_graph) {
  auto gpu = resource_manager_->get_local_gpu(raw_device_id);
  const auto flags = from_graph ? cudaEventWaitExternal : 0;
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream, gpu->get_event("sparse_tensors_ready"), flags));
}

template <typename SparseType>
void AsyncReader<SparseType>::stream_wait_dense_tensors(cudaStream_t stream, int raw_device_id,
                                                        bool from_graph) {
  auto gpu = resource_manager_->get_local_gpu(raw_device_id);
  const auto flags = from_graph ? cudaEventWaitExternal : 0;
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream, gpu->get_event("bottom_MLP_tensors_ready"), flags));
}

template <typename SparseType>
bool AsyncReader<SparseType>::current_batch_incomplete() const {
  return current_batch_size_ != batch_size_;
}

template <typename SparseType>
void AsyncReader<SparseType>::ready_to_collect() {
  auto raw_device_id = reader_impl_->get_last_batch_device();
  auto local_gpu = resource_manager_->get_local_gpu(raw_device_id);
  CudaDeviceContext ctx(local_gpu->get_device_id());

  reader_impl_->finalize_batch(&completion_events_[raw_device_id]);
}

template <typename SparseType>
long long AsyncReader<SparseType>::read_a_batch_to_device() {
  auto result = read_a_batch_to_device_delay_release();
  ready_to_collect();
  return result;
}

template <typename SparseType>
void AsyncReader<SparseType>::schedule_split_3_way_here(cudaStream_t stream, int raw_device_id,
                                                        bool from_graph) {
  unsigned int flags = from_graph ? cudaEventRecordExternal : 0;
  HCTR_LIB_THROW(cudaEventRecordWithFlags(split_schedule_events_[raw_device_id], stream, flags));
}

template <typename SparseType>
void AsyncReader<SparseType>::schedule_d2d_here(cudaStream_t stream, int raw_device_id,
                                                bool from_graph) {
  unsigned int flags = from_graph ? cudaEventRecordExternal : 0;
  HCTR_LIB_THROW(cudaEventRecordWithFlags(d2d_schedule_events_[raw_device_id], stream, flags));
}

template <typename SparseType>
void AsyncReader<SparseType>::schedule_here(cudaStream_t stream, int raw_device_id) {
  HCTR_LIB_THROW(cudaEventRecord(schedule_events_[raw_device_id], stream));
  reader_impl_->wait_for_gpu_event(&schedule_events_[raw_device_id], raw_device_id);
}

template <typename SparseType>
void AsyncReader<SparseType>::schedule_here_graph(cudaStream_t stream, int raw_device_id) {
  HCTR_LIB_THROW(
      cudaEventRecordWithFlags(schedule_events_[raw_device_id], stream, cudaEventRecordExternal));
}

template <typename SparseType>
void AsyncReader<SparseType>::update_schedule_graph(int raw_device_id) {
  reader_impl_->wait_for_gpu_event(&schedule_events_[raw_device_id], raw_device_id);
}

template <typename SparseType>
size_t AsyncReader<SparseType>::get_max_batches_inflight() const {
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

template <typename SparseType>
SparseTensors<SparseType> AsyncReader<SparseType>::get_value_tensors() const {
  SparseTensors<SparseType> tmp_tensors;
  tmp_tensors.reserve(current_sparse_tensors_.size());
  for (auto& st : current_sparse_tensors_) {
    tmp_tensors.emplace_back(SparseTensor<SparseType>::stretch_from(st));
  }
  return tmp_tensors;
}

template <typename SparseType>
std::vector<std::vector<SparseTensor<SparseType>>>
AsyncReader<SparseType>::get_value_tensor_buffers() const {
  std::vector<std::vector<SparseTensor<SparseType>>> tensors;
  for (const auto& batch_tensor : inflight_batch_tensors_) {
    std::vector<SparseTensor<SparseType>> gpu_tensors;
    for (const auto& sparse_tensor : batch_tensor.sparse_tensors) {
      gpu_tensors.emplace_back(SparseTensor<SparseType>::stretch_from(sparse_tensor));
    }
    tensors.emplace_back(gpu_tensors);
  }
  return tensors;
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
                                                  bool strict_order_of_batches,
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

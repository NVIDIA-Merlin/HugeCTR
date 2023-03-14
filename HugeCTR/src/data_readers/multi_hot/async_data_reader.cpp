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
#include <core23/tensor.hpp>
#include <data_reader.hpp>
#include <data_readers/async_reader/async_reader_adapter.hpp>
#include <data_readers/async_reader/async_reader_common.hpp>
#include <data_readers/multi_hot/async_data_reader.hpp>
#include <data_readers/multi_hot/split_batch.hpp>
#include <inference/preallocated_buffer2.hpp>
#include <resource_manager.hpp>
#include <tensor2.hpp>
#include <utils.hpp>

namespace HugeCTR {
namespace MultiHot {
namespace core23_reader {
template <typename SparseType>
AsyncDataReader<SparseType>::AsyncDataReader(
    std::vector<FileSource> data_files, const std::shared_ptr<ResourceManager>& resource_manager,
    size_t batch_size, size_t num_threads_per_file, size_t num_batches_per_thread,
    const std::vector<DataReaderSparseParam>& params, size_t label_dim, size_t dense_dim,
    bool mixed_precision, bool shuffle, bool schedule_uploads, bool is_dense_float)
    : resource_manager_(resource_manager),
      mixed_precision_(mixed_precision),
      batch_size_(batch_size),
      batch_size_per_dev_(batch_size / resource_manager->get_global_gpu_count()),
      completion_events_(resource_manager->get_local_gpu_count()),
      schedule_events_(resource_manager->get_local_gpu_count()),
      split_schedule_events_(resource_manager->get_local_gpu_count()),
      d2d_schedule_events_(resource_manager->get_local_gpu_count()),
      s3w_streams_(resource_manager->get_local_gpu_count()),
      d2d_streams_(resource_manager->get_local_gpu_count()),
      cache_buffers_(false),
      is_dense_float_(is_dense_float) {
  assert(batch_size_ % resource_manager_->get_global_gpu_count() == 0);
  assert(params.size() == 1);
  static_assert(sizeof(LabelType) == sizeof(InputType));

  size_t dense_dim_align8 = dense_dim;

  nnz_per_slot_ = params[0].nnz_per_slot;
  total_nnz_ = std::accumulate(nnz_per_slot_.begin(), nnz_per_slot_.end(), 0ull);

  std::vector<int> bucket_ids;
  std::vector<int> bucket_positions(static_cast<int64_t>(total_nnz_));
  int bucket = 0;
  auto bucket_begin = bucket_positions.begin();
  for (auto hotness : nnz_per_slot_) {
    bucket_ids.insert(bucket_ids.end(), hotness, bucket);
    bucket++;

    std::iota(bucket_begin, bucket_begin + hotness, 0);
    std::advance(bucket_begin, hotness);
  }

  size_t sparse_dim = params[0].slot_num;

  sample_size_items_ = label_dim + dense_dim +
                       static_cast<int64_t>(total_nnz_) * (sizeof(SparseType) / sizeof(InputType));

  label_dim_ = label_dim;
  dense_dim_ = dense_dim_align8;
  sparse_dim_ = sparse_dim;

  data_files[0].sample_size_bytes = sample_size_items_ * sizeof(InputType);

  reader_impl_.reset(new DataReaderImpl(data_files, resource_manager, batch_size,
                                        num_threads_per_file, num_batches_per_thread, shuffle,
                                        schedule_uploads));

  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    auto local_gpu = resource_manager_->get_local_gpu(i);
    auto gpu_id = local_gpu->get_device_id();
    CudaDeviceContext ctx(gpu_id);
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&completion_events_[i], cudaEventDisableTiming));
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&schedule_events_[i], cudaEventDisableTiming));
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&split_schedule_events_[i], cudaEventDisableTiming));
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&d2d_schedule_events_[i], cudaEventDisableTiming));

    core23::Tensor bucket_id_tensor(
        core23::TensorParams()
            .shape({static_cast<int64_t>(total_nnz_)})
            .data_type(core23::ScalarType::Int32)
            .device({core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)}));
    core23::Tensor bucket_position_tensor(
        core23::TensorParams()
            .shape({static_cast<int64_t>(total_nnz_)})
            .data_type(core23::ScalarType::Int32)
            .device({core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)}));
    core23::Tensor max_hotness_tensor(
        core23::TensorParams()
            .shape({static_cast<int64_t>(sparse_dim_)})
            .data_type(core23::ScalarType::Int32)
            .device({core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)}));

    // Create mapping from sparse column to bucket. This is used for split_3_way_feat_major
    HCTR_LIB_THROW(cudaMemcpy(bucket_id_tensor.data(), bucket_ids.data(),
                              static_cast<int64_t>(total_nnz_) * sizeof(int),
                              cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(bucket_position_tensor.data(), bucket_positions.data(),
                              static_cast<int64_t>(total_nnz_) * sizeof(int),
                              cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(max_hotness_tensor.data(), nnz_per_slot_.data(),
                              sparse_dim_ * sizeof(int), cudaMemcpyHostToDevice));

    bucket_id_tensors_.emplace_back(bucket_id_tensor);
    bucket_position_tensors_.emplace_back(bucket_position_tensor);
    max_hotness_tensors_.emplace_back(max_hotness_tensor);

    // set default stream
    s3w_streams_[i] = local_gpu->get_stream();
    d2d_streams_[i] = local_gpu->get_stream();
    int64_t bytes = batch_size_per_dev_ *
                    (label_dim * sizeof(LabelType) +
                     dense_dim_align8 * (mixed_precision ? sizeof(__half) : sizeof(float)));

    core23::Tensor one_tensor(
        core23::TensorParams()
            .device(core23::Device(core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)))
            .data_type(core23::ScalarType::Char)
            .shape({bytes}));
    temp_tensors_.push_back(one_tensor);

    label_tensors_.emplace_back(core23::Tensor::bind(
        one_tensor.data(),
        {static_cast<int64_t>(batch_size_per_dev_), static_cast<int64_t>(label_dim)},
        core23::ToScalarType<LabelType>::value,
        core23::Device(core23::DeviceType::GPU, static_cast<int8_t>(gpu_id))));

    dense_tensors_.emplace_back(core23::Tensor::bind(
        one_tensor.data<LabelType>() + batch_size_per_dev_ * label_dim,
        {static_cast<int64_t>(batch_size_per_dev_), static_cast<int64_t>(dense_dim_align8)},
        mixed_precision_ ? core23::ScalarType::Half : core23::ScalarType::Float,
        core23::Device(core23::DeviceType::GPU, static_cast<int8_t>(gpu_id))));
  }

  // zero-initialization
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    const auto local_gpu = resource_manager_->get_local_gpu(i);
    CudaDeviceContext ctx(local_gpu->get_device_id());
    core23::zeros_sync(dense_tensors_[i]);
  }

  set_tensor_buffering(1);
}

template <typename SparseType>
void AsyncDataReader<SparseType>::set_tensor_buffering(size_t num_batches_to_buffer) {
  // If the number of buffers exceeds or is equal to number of batches in our dataset, then we
  // may as well cache them so we only execute the 'split_3_way' kernel once.
  // cache_buffers_ = num_batches_to_buffer >= reader_impl_->get_total_batches();
  init_batch_tensors(num_batches_to_buffer);
}

template <typename SparseType>
void AsyncDataReader<SparseType>::init_batch_tensors(size_t num_inflight) {
  inflight_batch_tensors_.resize(num_inflight);

  for (auto& batch_tensors : inflight_batch_tensors_) {
    batch_tensors.tag = SIZE_MAX;  // Invalid

    for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
      auto local_gpu = resource_manager_->get_local_gpu(i);
      auto gpu_id = local_gpu->get_device_id();
      CudaDeviceContext ctx(gpu_id);
      int64_t bytes =
          batch_size_per_dev_ * (label_dim_ * sizeof(LabelType) +
                                 dense_dim_ * (mixed_precision_ ? sizeof(__half) : sizeof(float)));
      core23::Tensor one_tensor(
          core23::TensorParams()
              .device(core23::Device(core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)))
              .data_type(core23::ScalarType::Char)
              .shape({bytes}));
      temp_tensors_.push_back(one_tensor);
      batch_tensors.label_tensors.push_back(core23::Tensor::bind(
          one_tensor.data(),
          {static_cast<int64_t>(batch_size_per_dev_), static_cast<int64_t>(label_dim_)},
          core23::ToScalarType<LabelType>::value,
          core23::Device(core23::DeviceType::GPU, static_cast<int8_t>(gpu_id))));

      batch_tensors.dense_tensors.emplace_back(core23::Tensor::bind(
          one_tensor.data<LabelType>() + batch_size_per_dev_ * label_dim_,
          {static_cast<int64_t>(batch_size_per_dev_), static_cast<int64_t>(dense_dim_)},
          mixed_precision_ ? core23::ScalarType::Half : core23::ScalarType::Float,
          core23::Device(core23::DeviceType::GPU, static_cast<int8_t>(gpu_id))));
      core23::zeros_sync(batch_tensors.dense_tensors.back());

      core23::Tensor temp_sparse_tensor_ptrs(
          core23::TensorParams()
              .shape({static_cast<int64_t>(sparse_dim_), 1ll})
              .data_type(core23::ScalarType::UInt64)
              .device({core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)}));
      // allocate eagerly
      temp_sparse_tensor_ptrs.data();

      // Allocate sparse tensor for each feature
      // std::vector<SparseTensor<SparseType>> device_sparse_tensors;
      std::vector<SparseTensor23> device_sparse_tensors;
      for (size_t fea_id = 0; fea_id < sparse_dim_; ++fea_id) {
        const int64_t hotness = nnz_per_slot_[fea_id];
        SparseTensor23 temp_sparse_tensor(
            {static_cast<int64_t>(batch_size_per_dev_), static_cast<int64_t>(hotness)},
            core23::ToScalarType<SparseType>::value, core23::ToScalarType<SparseType>::value,
            hotness, {core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)});
        device_sparse_tensors.push_back(temp_sparse_tensor);
      }

      // Initialize sparse tensors
      for (size_t fea_id = 0; fea_id < sparse_dim_; ++fea_id) {
        auto value_ptr = device_sparse_tensors[fea_id].get_value_ptr();
        HCTR_LIB_THROW(
            cudaMemcpy(reinterpret_cast<SparseType**>(temp_sparse_tensor_ptrs.data()) + fea_id,
                       &value_ptr, sizeof(SparseType*), cudaMemcpyHostToDevice));

        const size_t hotness = device_sparse_tensors[fea_id].get_value_tensor().size(1);
        auto n = static_cast<SparseType>(0);
        std::vector<SparseType> row_offsets(batch_size_per_dev_ + 1);
        std::generate(row_offsets.begin(), row_offsets.end(),
                      [&n, hotness] { return n += hotness; });
        HCTR_LIB_THROW(cudaMemcpy(device_sparse_tensors[fea_id].get_rowoffset_ptr(),
                                  row_offsets.data(), row_offsets.size() * sizeof(SparseType),
                                  cudaMemcpyHostToDevice));

        *device_sparse_tensors[fea_id].get_nnz_ptr() = batch_size_per_dev_ * hotness;
      }

      batch_tensors.sparse_tensors.emplace_back(device_sparse_tensors);
      batch_tensors.sparse_tensor_ptrs.emplace_back(temp_sparse_tensor_ptrs);
    }
  }

  // Needed for get_value_tensors() on construction
  current_sparse_tensors_ = inflight_batch_tensors_.at(0).sparse_tensors;
}

template <typename SparseType>
long long AsyncDataReader<SparseType>::read_a_batch_to_device_delay_release() {
  const DataReaderImpl::Batch& batch = reader_impl_->get_batch();

  const size_t slot_id = 0;  // TODO: multi-hot

  if (cache_buffers_) {
    // TODO: replace with cache policy like LRU when number of batches exceeds what we can store
    inflight_id_ = batch.get_id();
  } else {
    inflight_id_ = (inflight_id_ + 1) % inflight_batch_tensors_.size();  // FIFO
  }

  BatchTensors& batch_tensors = inflight_batch_tensors_.at(inflight_id_);

  size_t current_batch_id = static_cast<size_t>(batch.get_id());
  current_batch_size_ = batch.get_batch_size_bytes() / (sample_size_items_ * sizeof(InputType));
  current_sparse_tensors_ = batch_tensors.sparse_tensors;
  current_batch_cached_ = (current_batch_id == batch_tensors.tag) && cache_buffers_;

  int num_local_gpus = resource_manager_->get_local_gpu_count();
#pragma omp parallel for num_threads(num_local_gpus)
  for (int i = 0; i < num_local_gpus; i++) {
    auto local_gpu = resource_manager_->get_local_gpu(i);
    auto gpu_id = local_gpu->get_device_id();
    CudaCPUDeviceContext ctx(gpu_id);

    const cudaStream_t& stream = s3w_streams_[i];

    size_t current_batch_size_per_device =
        batch.get_local_batch_size_bytes(i, slot_id) / (sample_size_items_ * sizeof(InputType));

    // schedule at correct place in iteration
    HCTR_LIB_THROW(cudaStreamWaitEvent(stream, split_schedule_events_[i]));

    if (!current_batch_cached_) {  // data can be cached for eval

      // >0 check because when batch is incomplete not all devices may have data-parallel shard
      if (static_cast<int64_t>(current_batch_size_per_device) > 0) {
        auto ptr_wrap = std::make_shared<RawPtrWrapper>(
            reinterpret_cast<InputType*>(batch.get_device_data(i, slot_id)));

        if (mixed_precision_) {
          split_3_way_feat_major<__half, SparseType>(
              batch_tensors.label_tensors[i], batch_tensors.dense_tensors[i],
              batch_tensors.sparse_tensor_ptrs[i],
              core23::Tensor::bind(
                  reinterpret_cast<void*>(ptr_wrap->get_ptr()),
                  {static_cast<int64_t>(current_batch_size_per_device),
                   static_cast<int64_t>(sample_size_items_)},
                  core23::ToScalarType<InputType>::value,
                  core23::Device(core23::DeviceType::GPU, static_cast<int8_t>(gpu_id))),
              bucket_id_tensors_[i], bucket_position_tensors_[i], max_hotness_tensors_[i], stream,
              is_dense_float_);
        } else {
          split_3_way_feat_major<float, SparseType>(
              batch_tensors.label_tensors[i], batch_tensors.dense_tensors[i],
              batch_tensors.sparse_tensor_ptrs[i],
              core23::Tensor::bind(
                  reinterpret_cast<void*>(ptr_wrap->get_ptr()),
                  {static_cast<int64_t>(current_batch_size_per_device),
                   static_cast<int64_t>(sample_size_items_)},
                  core23::ToScalarType<InputType>::value,
                  core23::Device(core23::DeviceType::GPU, static_cast<int8_t>(gpu_id))),
              bucket_id_tensors_[i], bucket_position_tensors_[i], max_hotness_tensors_[i], stream,
              is_dense_float_);
        }
      }
    }

    auto sparse_ready_event = local_gpu->get_event("sparse_tensors_ready");
    HCTR_LIB_THROW(cudaEventRecord(sparse_ready_event, stream));

    auto d2d_stream = d2d_streams_[i];

    // Need result from split-3-way
    HCTR_LIB_THROW(cudaStreamWaitEvent(d2d_stream, sparse_ready_event));

    // we are safe to overwrite
    HCTR_LIB_THROW(cudaStreamWaitEvent(d2d_stream, d2d_schedule_events_[i]));

    // isn't part of hybrid embedding
    assign_dense_and_label_tensors(batch_tensors.label_tensors[i], batch_tensors.dense_tensors[i],
                                   i, d2d_stream);

    auto tensors_ready_event = local_gpu->get_event("bottom_MLP_tensors_ready");
    HCTR_LIB_THROW(cudaEventRecord(tensors_ready_event, d2d_stream));

    // batch.device_data can be reused. Needs to be called after D2D because cudaStreamAddCallback
    // has latency and will delay execution of D2D.
    reader_impl_->device_release_last_batch_here(d2d_stream);
  }

  batch_tensors.tag = current_batch_id;
  return current_batch_size_;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::set_schedule_streams(cudaStream_t s3w_stream,
                                                       cudaStream_t d2d_stream, int raw_device_id) {
  s3w_streams_[raw_device_id] = s3w_stream;
  d2d_streams_[raw_device_id] = d2d_stream;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::assign_dense_and_label_tensors(core23::Tensor label_tensor,
                                                                 core23::Tensor dense_tensor,
                                                                 int raw_device_id,
                                                                 cudaStream_t stream) {
  auto& dst_label_tensor = label_tensors_[raw_device_id];
  auto& dst_dense_tensor = dense_tensors_[raw_device_id];

  // TODO: allocate tensors together
  if ((char*)dst_label_tensor.data() + dst_label_tensor.num_bytes() ==
      (char*)dst_dense_tensor.data()) {
    HCTR_LIB_THROW(cudaMemcpyAsync(dst_label_tensor.data(), label_tensor.data(),
                                   dst_label_tensor.num_bytes() + dense_tensor.num_bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
  } else {
    HCTR_LIB_THROW(cudaMemcpyAsync(dst_label_tensor.data(), label_tensor.data(),
                                   dst_label_tensor.num_bytes(), cudaMemcpyDeviceToDevice, stream));

    HCTR_LIB_THROW(cudaMemcpyAsync(dst_dense_tensor.data(), dense_tensor.data(),
                                   dst_dense_tensor.num_bytes(), cudaMemcpyDeviceToDevice, stream));
  }
}

template <typename SparseType>
long long AsyncDataReader<SparseType>::get_full_batchsize() const {
  return batch_size_;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::stream_wait_sparse_tensors(cudaStream_t stream, int raw_device_id,
                                                             bool from_graph) {
  auto gpu = resource_manager_->get_local_gpu(raw_device_id);
  const auto flags = from_graph ? cudaEventWaitExternal : 0;
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream, gpu->get_event("sparse_tensors_ready"), flags));
}

template <typename SparseType>
void AsyncDataReader<SparseType>::stream_wait_dense_tensors(cudaStream_t stream, int raw_device_id,
                                                            bool from_graph) {
  auto gpu = resource_manager_->get_local_gpu(raw_device_id);
  const auto flags = from_graph ? cudaEventWaitExternal : 0;
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream, gpu->get_event("bottom_MLP_tensors_ready"), flags));
}

template <typename SparseType>
bool AsyncDataReader<SparseType>::current_batch_incomplete() const {
  return current_batch_size_ != batch_size_;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::ready_to_collect() {
  // nothing to do, already released
}

template <typename SparseType>
long long AsyncDataReader<SparseType>::read_a_batch_to_device() {
  auto result = read_a_batch_to_device_delay_release();
  ready_to_collect();
  return result;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::schedule_split_3_way_here(cudaStream_t stream, int raw_device_id,
                                                            bool from_graph) {
  unsigned int flags = from_graph ? cudaEventRecordExternal : 0;
  HCTR_LIB_THROW(cudaEventRecordWithFlags(split_schedule_events_[raw_device_id], stream, flags));
}

template <typename SparseType>
void AsyncDataReader<SparseType>::schedule_d2d_here(cudaStream_t stream, int raw_device_id,
                                                    bool from_graph) {
  unsigned int flags = from_graph ? cudaEventRecordExternal : 0;
  HCTR_LIB_THROW(cudaEventRecordWithFlags(d2d_schedule_events_[raw_device_id], stream, flags));
}

template <typename SparseType>
void AsyncDataReader<SparseType>::schedule_here(cudaStream_t stream, int raw_device_id) {
  reader_impl_->schedule_upload_here(raw_device_id, stream, false);
}

template <typename SparseType>
void AsyncDataReader<SparseType>::schedule_here_graph(cudaStream_t stream, int raw_device_id) {
  reader_impl_->schedule_upload_here(raw_device_id, stream, true);
}

template <typename SparseType>
void AsyncDataReader<SparseType>::update_schedule_graph(int raw_device_id) {
  reader_impl_->upload_notify(raw_device_id);
}

template <typename SparseType>
size_t AsyncDataReader<SparseType>::get_max_batches_inflight() const {
  return reader_impl_->get_total_inflight_batches();
}

template <typename SparseType>
bool AsyncDataReader<SparseType>::is_mixed_precision() {
  return mixed_precision_;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::get_dimensions(size_t& label_dim, size_t& dense_dim,
                                                 size_t& sparse_dim, size_t& sample_size_items) {
  label_dim = label_dim_;
  dense_dim = dense_dim_;
  sparse_dim = sparse_dim_;
  sample_size_items = sample_size_items_;
}

template <typename SparseType>
long long AsyncDataReader<SparseType>::get_current_batchsize_per_device(size_t local_id) {
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
TensorScalarType AsyncDataReader<SparseType>::get_scalar_type() const {
  return TensorScalarTypeFunc<SparseType>::get_type();
};
template <typename SparseType>
bool AsyncDataReader<SparseType>::is_started() const {
  return true;
  // return reader_impl_->is_currently_loading();
}
template <typename SparseType>
void AsyncDataReader<SparseType>::start() {
  reader_impl_->start();
}
// TODO FIXME
template <typename SparseType>
std::vector<TensorBag2> AsyncDataReader<SparseType>::get_label_tensors() const {
  // LabelType = float
  std::vector<Tensor2<float>> ret;
  // convert tensor to tensor2
  for (auto tensor : label_tensors_) {
    // std::shared_ptr<Tensor2<float>> tensor2(new Tensor2<float>);
    auto shape = tensor.shape();
    std::vector<size_t> dimensions(shape.data(), shape.data() + shape.dims());
    auto buffer = PreallocatedBuffer2<float>::create(tensor.data(), dimensions);
    Tensor2<float> tensor2(dimensions, buffer);
    // bind_tensor_to_buffer(dimensions, buffer, tensor2);
    ret.push_back(tensor2);
  }
  // return label_tensors_;
  return tensors_to_bags(ret);
}
template <typename SparseType>
std::vector<TensorBag2> AsyncDataReader<SparseType>::get_dense_tensors() const {
  if (mixed_precision_) {
    std::vector<Tensor2<__half>> ret;
    for (auto tensor : dense_tensors_) {
      auto shape = tensor.shape();
      std::vector<size_t> dimensions(shape.data(), shape.data() + shape.dims());
      auto buffer = PreallocatedBuffer2<__half>::create(tensor.data(), dimensions);
      Tensor2<__half> tensor2(dimensions, buffer);
      ret.push_back(tensor2);
    }
    return tensors_to_bags(ret);
  } else {
    std::vector<Tensor2<float>> ret;
    for (auto tensor : dense_tensors_) {
      auto shape = tensor.shape();
      std::vector<size_t> dimensions(shape.data(), shape.data() + shape.dims());
      auto buffer = PreallocatedBuffer2<float>::create(tensor.data(), dimensions);
      Tensor2<float> tensor2(dimensions, buffer);
      ret.push_back(tensor2);
    }
    return tensors_to_bags(ret);
  }
  // return dense_tensors_;
  return {};
}
template <typename SparseType>
SparseTensors<SparseType> AsyncDataReader<SparseType>::get_value_tensors() const {
  throw std::runtime_error("Deprecated");
  return {};
}
// TODO FIXME
template <typename SparseType>
std::vector<std::vector<SparseTensor<SparseType>>>
AsyncDataReader<SparseType>::get_current_sparse_tensors() const {
  std::vector<std::vector<SparseTensor<SparseType>>> ret;

  for (auto sparse23s : current_sparse_tensors_) {
    std::vector<SparseTensor<SparseType>> sparse_tensors_fields;

    for (auto sparse23 : sparse23s) {
      auto value_tensor = sparse23.get_value_tensor();
      auto off_tensor = sparse23.get_rowoffset_tensor();
      auto shape = value_tensor.shape();
      std::shared_ptr<size_t> nnz_ptr = std::make_shared<size_t>(sparse23.nnz());
      // size_t* nnz_raw_ptr = reinterpret_cast<size_t*>(sparse23.get_nnz_ptr().get());
      std::vector<size_t> val_dimensions(shape.data(), shape.data() + shape.dims());
      auto val_buffer =
          PreallocatedBuffer2<SparseType>::create(value_tensor.data(), val_dimensions);

      shape = off_tensor.shape();
      std::vector<size_t> off_dimensions(shape.data(), shape.data() + shape.dims());
      auto off_buffer = PreallocatedBuffer2<SparseType>::create(off_tensor.data(), off_dimensions);
      SparseTensor<SparseType> sparse_tensor(val_dimensions, val_buffer, off_buffer, nnz_ptr,
                                             off_tensor.num_elements());
      sparse_tensors_fields.push_back(sparse_tensor);
    }
    ret.push_back(sparse_tensors_fields);
  }
  return ret;
}

template <typename SparseType>
std::vector<std::vector<SparseTensor<SparseType>>>
AsyncDataReader<SparseType>::get_value_tensor_buffers() const {
  throw std::runtime_error("Deprecated");
  return {};
}

template <typename SparseType>
void AsyncDataReader<SparseType>::create_drwg_norm(std::string file_list, Check_t check_type,
                                                   bool start_reading_from_beginning) {}
template <typename SparseType>
void AsyncDataReader<SparseType>::create_drwg_raw(std::string file_name, long long num_samples,
                                                  bool float_label_dense, bool data_shuffle,
                                                  bool start_reading_from_beginning) {}
#ifndef DISABLE_CUDF
template <typename SparseType>
void AsyncDataReader<SparseType>::create_drwg_parquet(std::string file_list,
                                                      bool strict_order_of_batches,
                                                      const std::vector<long long> slot_offset,
                                                      bool start_reading_from_beginning,
                                                      long long max_samples_per_group,
                                                      int label_dense_num, int label_dense_dim) {}
#endif
template <typename SparseType>
void AsyncDataReader<SparseType>::set_source(std::string file_list) {}

template <typename SparseType>
AsyncDataReader<SparseType>::~AsyncDataReader() {
  // Underlying reader mush be destroyed BEFORE the events
  reader_impl_.reset(nullptr);
  for (auto& e : completion_events_) {
    cudaEventDestroy(e);
  }
  for (auto& e : schedule_events_) {
    cudaEventDestroy(e);
  }
}

};  // namespace core23_reader

template <typename SparseType>
AsyncDataReader<SparseType>::AsyncDataReader(
    std::vector<FileSource> data_files, const std::shared_ptr<ResourceManager>& resource_manager,
    size_t batch_size, size_t num_threads_per_file, size_t num_batches_per_thread,
    const std::vector<DataReaderSparseParam>& params, size_t label_dim, size_t dense_dim,
    bool mixed_precision, bool shuffle, bool schedule_uploads, bool is_dense_float)
    : resource_manager_(resource_manager),
      mixed_precision_(mixed_precision),
      batch_size_(batch_size),
      batch_size_per_dev_(batch_size / resource_manager->get_global_gpu_count()),
      completion_events_(resource_manager->get_local_gpu_count()),
      schedule_events_(resource_manager->get_local_gpu_count()),
      split_schedule_events_(resource_manager->get_local_gpu_count()),
      d2d_schedule_events_(resource_manager->get_local_gpu_count()),
      s3w_streams_(resource_manager->get_local_gpu_count()),
      d2d_streams_(resource_manager->get_local_gpu_count()),
      cache_buffers_(false),
      is_dense_float_(is_dense_float) {
  assert(batch_size_ % resource_manager_->get_global_gpu_count() == 0);
  assert(params.size() == 1);
  static_assert(sizeof(LabelType) == sizeof(InputType));

  size_t dense_dim_align8 = dense_dim;

  // We need to align the dense dimension to prevent misaligned reads/writes when using half
  // precision in split_3_way
  //    if (false /*aligned == Alignment_t::Auto*/) {
  //      dense_dim_align8 = (dense_dim + 7) / 8 * 8;
  //    }

  nnz_per_slot_ = params[0].nnz_per_slot;
  total_nnz_ = std::accumulate(nnz_per_slot_.begin(), nnz_per_slot_.end(), 0);

  std::vector<int> bucket_ids;
  std::vector<int> bucket_positions(total_nnz_);
  int bucket = 0;
  auto bucket_begin = bucket_positions.begin();
  for (auto hotness : nnz_per_slot_) {
    bucket_ids.insert(bucket_ids.end(), hotness, bucket);
    bucket++;

    std::iota(bucket_begin, bucket_begin + hotness, 0);
    std::advance(bucket_begin, hotness);
  }

  //  printf("bucket_ids: ");
  //  for (auto id : bucket_ids) {
  //    printf("%d ", id);
  //  }
  //  printf("\nbucket_positions: ");
  //  for (auto pos : bucket_positions) {
  //    printf("%d ", pos);
  //  }
  //
  //  printf("\ntotal_nnz: %zu\n", bucket_ids.size());

  size_t sparse_dim = params[0].slot_num;

  sample_size_items_ =
      label_dim + dense_dim + total_nnz_ * (sizeof(SparseType) / sizeof(InputType));

  label_dim_ = label_dim;
  dense_dim_ = dense_dim_align8;
  sparse_dim_ = sparse_dim;

  data_files[0].sample_size_bytes = sample_size_items_ * sizeof(InputType);

  reader_impl_.reset(new DataReaderImpl(data_files, resource_manager, batch_size,
                                        num_threads_per_file, num_batches_per_thread, shuffle,
                                        schedule_uploads));

  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    auto local_gpu = resource_manager_->get_local_gpu(i);
    CudaDeviceContext ctx(local_gpu->get_device_id());
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&completion_events_[i], cudaEventDisableTiming));
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&schedule_events_[i], cudaEventDisableTiming));
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&split_schedule_events_[i], cudaEventDisableTiming));
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&d2d_schedule_events_[i], cudaEventDisableTiming));

    auto allocator = GeneralBuffer2<CudaAllocator>::create();

    Tensor2<int> bucket_id_tensor;
    Tensor2<int> bucket_position_tensor;
    Tensor2<int> max_hotness_tensor;

    allocator->reserve({total_nnz_}, &bucket_id_tensor);
    allocator->reserve({total_nnz_}, &bucket_position_tensor);
    allocator->reserve({sparse_dim_}, &max_hotness_tensor);
    allocator->allocate();

    // Create mapping from sparse column to bucket. This is used for split_3_way_feat_major
    HCTR_LIB_THROW(cudaMemcpy(bucket_id_tensor.get_ptr(), bucket_ids.data(),
                              total_nnz_ * sizeof(int), cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(bucket_position_tensor.get_ptr(), bucket_positions.data(),
                              total_nnz_ * sizeof(int), cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(max_hotness_tensor.get_ptr(), nnz_per_slot_.data(),
                              sparse_dim_ * sizeof(int), cudaMemcpyHostToDevice));

    bucket_id_tensors_.emplace_back(bucket_id_tensor);
    bucket_position_tensors_.emplace_back(bucket_position_tensor);
    max_hotness_tensors_.emplace_back(max_hotness_tensor);

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
    if (mixed_precision_) {
      Tensor2<__half> tensor = Tensor2<__half>::stretch_from(dense_tensors_[i]);
      HCTR_LIB_THROW(cudaMemsetAsync(tensor.get_ptr(), 0,
                                     tensor.get_num_elements() * sizeof(__half),
                                     local_gpu->get_memcpy_stream()));
    } else {
      Tensor2<float> tensor = Tensor2<float>::stretch_from(dense_tensors_[i]);
      HCTR_LIB_THROW(cudaMemsetAsync(tensor.get_ptr(), 0, tensor.get_num_elements() * sizeof(float),
                                     local_gpu->get_memcpy_stream()));
    }
  }

  set_tensor_buffering(1);
}

template <typename SparseType>
void AsyncDataReader<SparseType>::set_tensor_buffering(size_t num_batches_to_buffer) {
  // If the number of buffers exceeds or is equal to number of batches in our dataset, then we
  // may as well cache them so we only execute the 'split_3_way' kernel once.
  // cache_buffers_ = num_batches_to_buffer >= reader_impl_->get_total_batches();
  init_batch_tensors(num_batches_to_buffer);
}

template <typename SparseType>
void AsyncDataReader<SparseType>::init_batch_tensors(size_t num_inflight) {
  inflight_batch_tensors_.resize(num_inflight);

  for (auto& batch_tensors : inflight_batch_tensors_) {
    batch_tensors.tag = SIZE_MAX;  // Invalid

    for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
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

      auto device_allocator = GeneralBuffer2<CudaAllocator>::create();

      Tensor2<SparseType*> temp_sparse_tensor_ptrs;
      device_allocator->reserve({sparse_dim_, 1}, &temp_sparse_tensor_ptrs);

      // Allocate sparse tensor for each feature
      std::vector<SparseTensor<SparseType>> device_sparse_tensors;
      for (size_t fea_id = 0; fea_id < sparse_dim_; ++fea_id) {
        SparseTensor<SparseType> temp_sparse_tensor;
        const size_t hotness = nnz_per_slot_[fea_id];
        device_allocator->reserve({batch_size_per_dev_, hotness}, hotness, &temp_sparse_tensor);
        device_sparse_tensors.push_back(temp_sparse_tensor);
      }
      device_allocator->allocate();

      // Initialize sparse tensors
      for (size_t fea_id = 0; fea_id < sparse_dim_; ++fea_id) {
        auto value_ptr = device_sparse_tensors[fea_id].get_value_ptr();
        HCTR_LIB_THROW(cudaMemcpy(temp_sparse_tensor_ptrs.get_ptr() + fea_id, &value_ptr,
                                  sizeof(SparseType*), cudaMemcpyHostToDevice));

        const size_t hotness = device_sparse_tensors[fea_id].get_dimensions()[1];
        auto n = static_cast<SparseType>(0);
        std::vector<SparseType> row_offsets(batch_size_per_dev_ + 1);
        std::generate(row_offsets.begin(), row_offsets.end(),
                      [&n, hotness] { return n += hotness; });
        HCTR_LIB_THROW(cudaMemcpy(device_sparse_tensors[fea_id].get_rowoffset_ptr(),
                                  row_offsets.data(), row_offsets.size() * sizeof(SparseType),
                                  cudaMemcpyHostToDevice));

        *device_sparse_tensors[fea_id].get_nnz_ptr() = batch_size_per_dev_ * hotness;
      }

      batch_tensors.sparse_tensors.emplace_back(device_sparse_tensors);
      batch_tensors.sparse_tensor_ptrs.emplace_back(temp_sparse_tensor_ptrs);
    }
  }

  // Needed for get_value_tensors() on construction
  current_sparse_tensors_ = inflight_batch_tensors_.at(0).sparse_tensors;
}

template <typename SparseType>
long long AsyncDataReader<SparseType>::read_a_batch_to_device_delay_release() {
  const DataReaderImpl::Batch& batch = reader_impl_->get_batch();

  const size_t slot_id = 0;  // TODO: multi-hot

  if (cache_buffers_) {
    // TODO: replace with cache policy like LRU when number of batches exceeds what we can store
    inflight_id_ = batch.get_id();
  } else {
    inflight_id_ = (inflight_id_ + 1) % inflight_batch_tensors_.size();  // FIFO
  }

  BatchTensors& batch_tensors = inflight_batch_tensors_.at(inflight_id_);

  size_t current_batch_id = static_cast<size_t>(batch.get_id());
  current_batch_size_ = batch.get_batch_size_bytes() / (sample_size_items_ * sizeof(InputType));
  current_sparse_tensors_ = batch_tensors.sparse_tensors;
  current_batch_cached_ = (current_batch_id == batch_tensors.tag) && cache_buffers_;

  int num_local_gpus = resource_manager_->get_local_gpu_count();
#pragma omp parallel for num_threads(num_local_gpus)
  for (int i = 0; i < num_local_gpus; i++) {
    auto local_gpu = resource_manager_->get_local_gpu(i);
    CudaCPUDeviceContext ctx(local_gpu->get_device_id());

    const cudaStream_t& stream = s3w_streams_[i];

    size_t current_batch_size_per_device =
        batch.get_local_batch_size_bytes(i, slot_id) / (sample_size_items_ * sizeof(InputType));

    // schedule at correct place in iteration
    HCTR_LIB_THROW(cudaStreamWaitEvent(stream, split_schedule_events_[i]));

    if (!current_batch_cached_) {  // data can be cached for eval

      // >0 check because when batch is incomplete not all devices may have data-parallel shard
      if (current_batch_size_per_device > 0) {
        auto ptr_wrap = std::make_shared<RawPtrWrapper>(
            reinterpret_cast<InputType*>(batch.get_device_data(i, slot_id)));

        if (mixed_precision_) {
          split_3_way_feat_major(
              Tensor2<LabelType>::stretch_from(batch_tensors.label_tensors[i]),
              Tensor2<__half>::stretch_from(batch_tensors.dense_tensors[i]),
              batch_tensors.sparse_tensor_ptrs[i],
              Tensor2<InputType>({current_batch_size_per_device, sample_size_items_}, ptr_wrap),
              bucket_id_tensors_[i], bucket_position_tensors_[i], max_hotness_tensors_[i], stream,
              is_dense_float_);
        } else {
          split_3_way_feat_major(
              Tensor2<LabelType>::stretch_from(batch_tensors.label_tensors[i]),
              Tensor2<float>::stretch_from(batch_tensors.dense_tensors[i]),
              batch_tensors.sparse_tensor_ptrs[i],
              Tensor2<InputType>({current_batch_size_per_device, sample_size_items_}, ptr_wrap),
              bucket_id_tensors_[i], bucket_position_tensors_[i], max_hotness_tensors_[i], stream,
              is_dense_float_);
        }
      }
    }

    auto sparse_ready_event = local_gpu->get_event("sparse_tensors_ready");
    HCTR_LIB_THROW(cudaEventRecord(sparse_ready_event, stream));

    auto d2d_stream = d2d_streams_[i];

    // Need result from split-3-way
    HCTR_LIB_THROW(cudaStreamWaitEvent(d2d_stream, sparse_ready_event));

    // we are safe to overwrite
    HCTR_LIB_THROW(cudaStreamWaitEvent(d2d_stream, d2d_schedule_events_[i]));

    // isn't part of hybrid embedding
    assign_dense_and_label_tensors(batch_tensors.label_tensors[i], batch_tensors.dense_tensors[i],
                                   i, d2d_stream);

    auto tensors_ready_event = local_gpu->get_event("bottom_MLP_tensors_ready");
    HCTR_LIB_THROW(cudaEventRecord(tensors_ready_event, d2d_stream));

    // batch.device_data can be reused. Needs to be called after D2D because cudaStreamAddCallback
    // has latency and will delay execution of D2D.
    reader_impl_->device_release_last_batch_here(d2d_stream);
  }

  batch_tensors.tag = current_batch_id;
  return current_batch_size_;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::set_schedule_streams(cudaStream_t s3w_stream,
                                                       cudaStream_t d2d_stream, int raw_device_id) {
  s3w_streams_[raw_device_id] = s3w_stream;
  d2d_streams_[raw_device_id] = d2d_stream;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::assign_dense_and_label_tensors(TensorBag2 label_tensor,
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
long long AsyncDataReader<SparseType>::get_full_batchsize() const {
  return batch_size_;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::stream_wait_sparse_tensors(cudaStream_t stream, int raw_device_id,
                                                             bool from_graph) {
  auto gpu = resource_manager_->get_local_gpu(raw_device_id);
  const auto flags = from_graph ? cudaEventWaitExternal : 0;
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream, gpu->get_event("sparse_tensors_ready"), flags));
}

template <typename SparseType>
void AsyncDataReader<SparseType>::stream_wait_dense_tensors(cudaStream_t stream, int raw_device_id,
                                                            bool from_graph) {
  auto gpu = resource_manager_->get_local_gpu(raw_device_id);
  const auto flags = from_graph ? cudaEventWaitExternal : 0;
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream, gpu->get_event("bottom_MLP_tensors_ready"), flags));
}

template <typename SparseType>
bool AsyncDataReader<SparseType>::current_batch_incomplete() const {
  return current_batch_size_ != batch_size_;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::ready_to_collect() {
  // nothing to do, already released
}

template <typename SparseType>
long long AsyncDataReader<SparseType>::read_a_batch_to_device() {
  auto result = read_a_batch_to_device_delay_release();
  ready_to_collect();
  return result;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::schedule_split_3_way_here(cudaStream_t stream, int raw_device_id,
                                                            bool from_graph) {
  unsigned int flags = from_graph ? cudaEventRecordExternal : 0;
  HCTR_LIB_THROW(cudaEventRecordWithFlags(split_schedule_events_[raw_device_id], stream, flags));
}

template <typename SparseType>
void AsyncDataReader<SparseType>::schedule_d2d_here(cudaStream_t stream, int raw_device_id,
                                                    bool from_graph) {
  unsigned int flags = from_graph ? cudaEventRecordExternal : 0;
  HCTR_LIB_THROW(cudaEventRecordWithFlags(d2d_schedule_events_[raw_device_id], stream, flags));
}

template <typename SparseType>
void AsyncDataReader<SparseType>::schedule_here(cudaStream_t stream, int raw_device_id) {
  reader_impl_->schedule_upload_here(raw_device_id, stream, false);
}

template <typename SparseType>
void AsyncDataReader<SparseType>::schedule_here_graph(cudaStream_t stream, int raw_device_id) {
  reader_impl_->schedule_upload_here(raw_device_id, stream, true);
}

template <typename SparseType>
void AsyncDataReader<SparseType>::update_schedule_graph(int raw_device_id) {
  reader_impl_->upload_notify(raw_device_id);
}

template <typename SparseType>
size_t AsyncDataReader<SparseType>::get_max_batches_inflight() const {
  return reader_impl_->get_total_inflight_batches();
}

template <typename SparseType>
bool AsyncDataReader<SparseType>::is_mixed_precision() {
  return mixed_precision_;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::get_dimensions(size_t& label_dim, size_t& dense_dim,
                                                 size_t& sparse_dim, size_t& sample_size_items) {
  label_dim = label_dim_;
  dense_dim = dense_dim_;
  sparse_dim = sparse_dim_;
  sample_size_items = sample_size_items_;
}

template <typename SparseType>
long long AsyncDataReader<SparseType>::get_current_batchsize_per_device(size_t local_id) {
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
TensorScalarType AsyncDataReader<SparseType>::get_scalar_type() const {
  return TensorScalarTypeFunc<SparseType>::get_type();
};
template <typename SparseType>
bool AsyncDataReader<SparseType>::is_started() const {
  return true;
  // return reader_impl_->is_currently_loading();
}
template <typename SparseType>
void AsyncDataReader<SparseType>::start() {
  reader_impl_->start();
}
template <typename SparseType>
std::vector<TensorBag2> AsyncDataReader<SparseType>::get_label_tensors() const {
  return label_tensors_;
}
template <typename SparseType>
std::vector<TensorBag2> AsyncDataReader<SparseType>::get_dense_tensors() const {
  return dense_tensors_;
}

template <typename SparseType>
SparseTensors<SparseType> AsyncDataReader<SparseType>::get_value_tensors() const {
  throw std::runtime_error("Deprecated");
  return {};
  //  SparseTensors<SparseType> tmp_tensors;
  //  tmp_tensors.reserve(current_sparse_tensors_.size());
  //  for (auto& st : current_sparse_tensors_) {
  //    tmp_tensors.emplace_back(SparseTensor<SparseType>::stretch_from(st));
  //  }
  //  return tmp_tensors;
}

template <typename SparseType>
std::vector<std::vector<SparseTensor<SparseType>>>
AsyncDataReader<SparseType>::get_current_sparse_tensors() const {
  return current_sparse_tensors_;
}

template <typename SparseType>
std::vector<std::vector<SparseTensor<SparseType>>>
AsyncDataReader<SparseType>::get_value_tensor_buffers() const {
  throw std::runtime_error("Deprecated");
  return {};
  //  std::vector<std::vector<SparseTensor<SparseType>>> tensors;
  //  for (const auto& batch_tensor : inflight_batch_tensors_) {
  //    std::vector<SparseTensor<SparseType>> gpu_tensors;
  //    for (const auto& sparse_tensor : batch_tensor.sparse_tensors) {
  //      gpu_tensors.emplace_back(SparseTensor<SparseType>::stretch_from(sparse_tensor));
  //    }
  //    tensors.emplace_back(gpu_tensors);
  //  }
  //  return tensors;
}

template <typename SparseType>
void AsyncDataReader<SparseType>::create_drwg_norm(std::string file_list, Check_t check_type,
                                                   bool start_reading_from_beginning) {}
template <typename SparseType>
void AsyncDataReader<SparseType>::create_drwg_raw(std::string file_name, long long num_samples,
                                                  bool float_label_dense, bool data_shuffle,
                                                  bool start_reading_from_beginning) {}
#ifndef DISABLE_CUDF
template <typename SparseType>
void AsyncDataReader<SparseType>::create_drwg_parquet(std::string file_list,
                                                      bool strict_order_of_batches,
                                                      const std::vector<long long> slot_offset,
                                                      bool start_reading_from_beginning,
                                                      long long max_samples_per_group,
                                                      int label_dense_num, int label_dense_dim) {}
#endif
template <typename SparseType>
void AsyncDataReader<SparseType>::set_source(std::string file_list) {}

template <typename SparseType>
AsyncDataReader<SparseType>::~AsyncDataReader() {
  // Underlying reader mush be destroyed BEFORE the events
  reader_impl_.reset(nullptr);
  for (auto& e : completion_events_) {
    cudaEventDestroy(e);
  }
  for (auto& e : schedule_events_) {
    cudaEventDestroy(e);
  }
  std::cout << "~AsyncDataReader" << std::endl;
}

template class AsyncDataReader<uint32_t>;
template class AsyncDataReader<long long>;
template class core23_reader::AsyncDataReader<uint32_t>;
template class core23_reader::AsyncDataReader<long long>;
}  // namespace MultiHot
}  // namespace HugeCTR

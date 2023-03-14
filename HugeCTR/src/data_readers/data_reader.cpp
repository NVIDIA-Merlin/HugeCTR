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
#include <data_readers/data_reader.hpp>

namespace HugeCTR {

namespace core23_reader {

template <typename TypeKey>
DataReader<TypeKey>::DataReader(int batchsize, size_t label_dim, int dense_dim,
                                std::vector<DataReaderSparseParam> &params,
                                const std::shared_ptr<ResourceManager> &resource_manager,
                                bool repeat, int num_threads, bool use_mixed_precision,
                                const DataSourceParams &data_source_params)
    : broadcast_buffer_(new BroadcastBuffer23()),
      output_(new DataReaderOutput()),
      params_(params),
      resource_manager_(resource_manager),
      batchsize_(batchsize),
      label_dim_(label_dim),
      dense_dim_(dense_dim),
      repeat_(repeat),
      data_source_params_(data_source_params) {
  auto row_offset_type = core23::ToScalarType<TypeKey>::value;
  CudaDeviceContext ctx;
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();
  size_t total_gpu_count = resource_manager_->get_global_gpu_count();
  // TODO REMOVE ME
  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> buffs;
  buffs.reserve(local_gpu_count);
  for (size_t i = 0; i < local_gpu_count; ++i) {
    buffs.push_back(GeneralBuffer2<CudaAllocator>::create());
  }
  // input check
  if (total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 ||
      0 != batchsize_ % total_gpu_count) {
    HCTR_OWN_THROW(
        Error_t::WrongInput,
        "total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 || 0 != "
        "batchsize_ % total_gpu_count");
  }
  // batchsize_ is a multiple of total_gpu_count
  size_t batch_size_per_gpu = batchsize_ / total_gpu_count;
  thread_buffers_.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // a worker may maintain multiple buffers on device i % local_gpu_count
    auto local_gpu = resource_manager_->get_local_gpu(i % local_gpu_count);
    auto gpu_id = local_gpu->get_device_id();
    CudaCPUDeviceContext context(gpu_id);
    std::shared_ptr<ThreadBuffer23> current_thread_buffer = std::make_shared<ThreadBuffer23>();
    thread_buffers_.push_back(current_thread_buffer);

    current_thread_buffer->device_sparse_buffers.reserve(params.size());
    current_thread_buffer->is_fixed_length.reserve(params.size());
    for (size_t param_id = 0; param_id < params.size(); ++param_id) {
      auto &param = params_[param_id];

      // to be compatible with
      current_thread_buffer->device_sparse_buffers.emplace_back(
          core23::Shape({(int64_t)batchsize, (int64_t)param.max_feature_num}),
          core23::ToScalarType<TypeKey>::value, row_offset_type, (int64_t)param.slot_num,
          core23::Device(core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)));
      // allocate eagerly
      current_thread_buffer->device_sparse_buffers.back().get_value_ptr();
      current_thread_buffer->device_sparse_buffers.back().get_rowoffset_ptr();
      current_thread_buffer->is_fixed_length.push_back(param.is_fixed_length);
    }

    current_thread_buffer->device_dense_buffers =
        core23::Tensor(core23::TensorParams()
                           .data_type(core23::ScalarType::Float)
                           .shape({(int64_t)(batch_size_per_gpu * local_gpu_count),
                                   (int64_t)(label_dim + dense_dim)})
                           .device({core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)}));
    // allocate eagerly
    current_thread_buffer->device_dense_buffers.data();

    current_thread_buffer->state.store(BufferState::ReadyForWrite);
    current_thread_buffer->current_batch_size = 0;
    current_thread_buffer->batch_size = batchsize;
    current_thread_buffer->param_num = params.size();
    current_thread_buffer->label_dim = label_dim;
    current_thread_buffer->dense_dim = dense_dim;
    current_thread_buffer->batch_size_start_idx =
        batch_size_per_gpu * resource_manager_->get_gpu_global_id_from_local_id(0);
    current_thread_buffer->batch_size_end_idx =
        current_thread_buffer->batch_size_start_idx + batch_size_per_gpu * local_gpu_count;
  }

  broadcast_buffer_->sparse_buffers.reserve(local_gpu_count * params.size());
  broadcast_buffer_->is_fixed_length.reserve(local_gpu_count * params.size());
  broadcast_buffer_->dense_tensors.reserve(local_gpu_count);
  broadcast_buffer_->finish_broadcast_events.resize(local_gpu_count);
  broadcast_buffer_->state.store(BufferState::ReadyForWrite);
  broadcast_buffer_->current_batch_size = 0;
  broadcast_buffer_->param_num = params.size();
  output_->dense_tensors.reserve(local_gpu_count);
  output_->label_tensors.reserve(local_gpu_count);
  output_->use_mixed_precision = use_mixed_precision;
  output_->label_dense_dim = label_dim + dense_dim;
  for (size_t param_id = 0; param_id < params.size(); ++param_id) {
    auto &param = params_[param_id];

    output_->sparse_tensors_map[param.top_name].reserve(local_gpu_count);
    output_->sparse_name_vec.push_back(param.top_name);
  }

  for (size_t local_id = 0; local_id < local_gpu_count; ++local_id) {
    auto local_gpu = resource_manager_->get_local_gpu(local_id);
    auto gpu_id = local_gpu->get_device_id();
    auto &buff = buffs[local_id];
    CudaDeviceContext ctx(gpu_id);

    for (size_t param_id = 0; param_id < params.size(); ++param_id) {
      auto &param = params_[param_id];
      broadcast_buffer_->sparse_buffers.emplace_back(
          core23::Shape({(int64_t)batchsize, (int64_t)param.max_feature_num}),
          core23::ToScalarType<TypeKey>::value, row_offset_type, (int64_t)param.slot_num,
          core23::Device(core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)));
      broadcast_buffer_->sparse_buffers.back().get_value_ptr();
      broadcast_buffer_->sparse_buffers.back().get_rowoffset_ptr();
      broadcast_buffer_->is_fixed_length.push_back(param.is_fixed_length);
    }
    auto temp_dense_tensor =
        core23::Tensor(core23::TensorParams()
                           .data_type(core23::ScalarType::Float)
                           .shape({(int64_t)(batch_size_per_gpu), (int64_t)(label_dim + dense_dim)})
                           .device({core23::DeviceType::GPU, static_cast<int8_t>(gpu_id)}));
    temp_dense_tensor.data();
    broadcast_buffer_->dense_tensors.push_back(temp_dense_tensor);

    HCTR_LIB_THROW(cudaEventCreateWithFlags(&broadcast_buffer_->finish_broadcast_events[local_id],
                                            cudaEventDisableTiming));
    // TODO FIXME with core23
    for (size_t param_id = 0; param_id < params.size(); ++param_id) {
      auto &param = params_[param_id];

      SparseTensor<TypeKey> temp_sparse_tensor;
      buff->reserve({(size_t)batchsize, (size_t)param.max_feature_num}, param.slot_num,
                    &temp_sparse_tensor);
      output_->sparse_tensors_map[param.top_name].push_back(temp_sparse_tensor.shrink());
    }

    // auto label_tensor = core23::Tensor(core23::TensorParams()
    //           .shape({(int64_t)batch_size_per_gpu, (int64_t)(label_dim) })
    //           .device( {core23::DeviceType::GPU,static_cast<int8_t>(gpu_id)})) ;
    // TODO FIXME
    Tensor2<float> label_tensor;
    buff->reserve({batch_size_per_gpu, label_dim}, &label_tensor);
    output_->label_tensors.push_back(label_tensor.shrink());

    if (use_mixed_precision) {
      Tensor2<__half> dense_tensor;
      buff->reserve({(size_t)batch_size_per_gpu, (size_t)dense_dim}, &dense_tensor);
      output_->dense_tensors.push_back(dense_tensor.shrink());
    } else {
      Tensor2<float> dense_tensor;
      buff->reserve({(size_t)batch_size_per_gpu, (size_t)dense_dim}, &dense_tensor);
      output_->dense_tensors.push_back(dense_tensor.shrink());
    }
    buff->allocate();
  }

  data_collector_ = std::make_shared<core23_reader::DataCollector<TypeKey>>(
      thread_buffers_, broadcast_buffer_, output_, resource_manager);
}
template <typename TypeKey>
DataReader<TypeKey>::~DataReader() {
  try {
    // stop all the loops
    data_collector_->stop();
    if (worker_group_) {
      worker_group_->end();
    }
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    for (size_t i = 0; i < local_gpu_count; ++i) {
      HCTR_LIB_THROW(cudaEventDestroy(broadcast_buffer_->finish_broadcast_events[i]));
    }
  } catch (const std::runtime_error &rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
}
template <typename TypeKey>
long long DataReader<TypeKey>::read_a_batch_to_device() {
  current_batchsize_ = read_a_batch_to_device_delay_release();
  ready_to_collect();
  return current_batchsize_;
}
template <typename TypeKey>
long long DataReader<TypeKey>::read_a_batch_to_device_delay_release() {
  current_batchsize_ = data_collector_->read_a_batch_to_device();
  return current_batchsize_;
}
template <typename TypeKey>
void DataReader<TypeKey>::ready_to_collect() {
  data_collector_->finalize_batch();
}

template <typename TypeKey>
long long DataReader<TypeKey>::get_current_batchsize_per_device(size_t local_id) {
  if (batchsize_ % resource_manager_->get_global_gpu_count() != 0) {
    HCTR_OWN_THROW(Error_t::UnspecificError,
                   "batchsize_ % resource_manager_->get_global_gpu_count() != 0");
  }
  long long batchsize_per_device = batchsize_ / resource_manager_->get_global_gpu_count();
  size_t global_id = resource_manager_->get_gpu_global_id_from_local_id(local_id);
  long long remain_samples = current_batchsize_ - global_id * batchsize_per_device;
  if (remain_samples >= batchsize_per_device) {
    return batchsize_per_device;
  } else if (remain_samples > 0) {
    return remain_samples;
  } else {
    return 0;
  }
}
template <typename TypeKey>
long long DataReader<TypeKey>::get_full_batchsize() const {
  return batchsize_;
}
template <typename TypeKey>
bool DataReader<TypeKey>::current_batch_incomplete() const {
  return current_batchsize_ != batchsize_;
  ;
}
template <typename TypeKey>
bool DataReader<TypeKey>::is_started() const {
  return worker_group_ && worker_group_->is_started();
}
template <typename TypeKey>
void DataReader<TypeKey>::start() {
  if (worker_group_ != nullptr) {
    worker_group_->start();
  } else {
    throw internal_runtime_error(Error_t::NotInitialized, "worker_group_ == nullptr");
  }
}
template <typename TypeKey>
const std::vector<SparseTensorBag> &DataReader<TypeKey>::get_sparse_tensors(
    const std::string &name) {
  if (output_->sparse_tensors_map.find(name) == output_->sparse_tensors_map.end()) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "no such sparse output in data reader:" + name);
  }
  return output_->sparse_tensors_map[name];
}
template <typename TypeKey>
const std::vector<TensorBag2> &DataReader<TypeKey>::get_label_tensors() const {
  return output_->label_tensors;
}
template <typename TypeKey>
const std::vector<TensorBag2> &DataReader<TypeKey>::get_dense_tensors() const {
  return output_->dense_tensors;
}
template <typename TypeKey>
const std::vector<SparseTensor23> &DataReader<TypeKey>::get_sparse_core23_tensors(
    const std::string &name) {
  if (output23_->sparse_tensors_map.find(name) == output23_->sparse_tensors_map.end()) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "no such sparse output in data reader:" + name);
  }
  return output23_->sparse_tensors_map[name];
}
template <typename TypeKey>
const std::vector<core23::Tensor> &DataReader<TypeKey>::get_label_core23_tensors23() const {
  return output23_->label_tensors;
}
template <typename TypeKey>
const std::vector<core23::Tensor> &DataReader<TypeKey>::get_dense_core23_tensors23() const {
  return output23_->dense_tensors;
}

template <typename TypeKey>
void DataReader<TypeKey>::set_source(std::string file_name) {
  if (worker_group_ != nullptr) {
    if (file_name.empty()) {
      if (file_name_.empty()) {
        throw internal_runtime_error(Error_t::NotInitialized, "invalid file_name");
      } else {
        file_name = file_name_;
      }
    }
    worker_group_->set_source(source_type_, file_name, repeat_, data_source_params_);
  } else {
    throw internal_runtime_error(Error_t::NotInitialized, "worker_group_ == nullptr");
  }
}
template <typename TypeKey>
void DataReader<TypeKey>::create_drwg_norm(std::string file_name, Check_t check_type,
                                           bool start_reading_from_beginning) {
  source_type_ = SourceType_t::FileList;
  worker_group_.reset(new core23_reader::DataReaderWorkerGroupNorm<TypeKey>(
      thread_buffers_, resource_manager_, file_name, repeat_, check_type, params_,
      start_reading_from_beginning));
  file_name_ = file_name;
}
template <typename TypeKey>
void DataReader<TypeKey>::create_drwg_raw(std::string file_name, long long num_samples,
                                          bool float_label_dense, bool data_shuffle,
                                          bool start_reading_from_beginning) {
  // check if key type compatible with dataset
  size_t file_size = std::filesystem::file_size(file_name);
  size_t expected_file_size = (label_dim_ + dense_dim_) * sizeof(float);
  for (auto &param : params_) {
    expected_file_size += param.slot_num * sizeof(TypeKey);
  }
  expected_file_size *= num_samples;
  if (file_size != expected_file_size) {
    HCTR_OWN_THROW(Error_t::UnspecificError,
                   "dataset key type and dataset size is not compatible.");
  }
  source_type_ = SourceType_t::Mmap;
  worker_group_.reset(new core23_reader::DataReaderWorkerGroupRaw<TypeKey>(
      thread_buffers_, resource_manager_, file_name, num_samples, repeat_, params_, label_dim_,
      dense_dim_, batchsize_, float_label_dense, data_shuffle, start_reading_from_beginning));
  file_name_ = file_name;
}
#ifndef DISABLE_CUDF
template <typename TypeKey>
void DataReader<TypeKey>::create_drwg_parquet(std::string file_list, bool strict_order_of_batches,
                                              const std::vector<long long> slot_offset,
                                              bool start_reading_from_beginning,
                                              long long max_samples_per_group, int label_dense_num,
                                              int label_dense_dim) {
  source_type_ = SourceType_t::Parquet;
  // worker_group_.empty
  worker_group_.reset(new core23_reader::DataReaderWorkerGroupParquet<TypeKey>(
      thread_buffers_, file_list, strict_order_of_batches, repeat_, params_, slot_offset,
      data_source_params_, resource_manager_, start_reading_from_beginning, label_dense_num,
      label_dense_dim, max_samples_per_group));
}
#endif
};  // namespace core23_reader

template <typename TypeKey>
DataReader<TypeKey>::DataReader(int batchsize, size_t label_dim, int dense_dim,
                                std::vector<DataReaderSparseParam> &params,
                                const std::shared_ptr<ResourceManager> &resource_manager,
                                bool repeat, int num_threads, bool use_mixed_precision,
                                const DataSourceParams &data_source_params)
    : broadcast_buffer_(new BroadcastBuffer()),
      output_(new DataReaderOutput()),
      params_(params),
      resource_manager_(resource_manager),
      batchsize_(batchsize),
      label_dim_(label_dim),
      dense_dim_(dense_dim),
      repeat_(repeat),
      data_source_params_(data_source_params) {
  CudaDeviceContext ctx;
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();
  size_t total_gpu_count = resource_manager_->get_global_gpu_count();

  // input check
  if (total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 ||
      0 != batchsize_ % total_gpu_count) {
    HCTR_OWN_THROW(
        Error_t::WrongInput,
        "total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 || 0 != "
        "batchsize_ % total_gpu_count");
  }
  // batchsize_ is a multiple of total_gpu_count
  size_t batch_size_per_gpu = batchsize_ / total_gpu_count;
  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> buffs;
  buffs.reserve(local_gpu_count);
  for (size_t i = 0; i < local_gpu_count; ++i) {
    buffs.push_back(GeneralBuffer2<CudaAllocator>::create());
  }
  thread_buffers_.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // a worker may maintain multiple buffers on device i % local_gpu_count
    auto local_gpu = resource_manager_->get_local_gpu(i % local_gpu_count);
    CudaCPUDeviceContext context(local_gpu->get_device_id());
    auto &buff = buffs[i % local_gpu_count];
    std::shared_ptr<ThreadBuffer> current_thread_buffer = std::make_shared<ThreadBuffer>();
    thread_buffers_.push_back(current_thread_buffer);

    current_thread_buffer->device_sparse_buffers.reserve(params.size());
    current_thread_buffer->is_fixed_length.reserve(params.size());
    for (size_t param_id = 0; param_id < params.size(); ++param_id) {
      auto &param = params_[param_id];
      SparseTensor<TypeKey> temp_sparse_tensor;
      buff->reserve({(size_t)batchsize, (size_t)param.max_feature_num}, param.slot_num,
                    &temp_sparse_tensor);
      current_thread_buffer->device_sparse_buffers.push_back(temp_sparse_tensor.shrink());

      current_thread_buffer->is_fixed_length.push_back(param.is_fixed_length);
    }
    Tensor2<float> temp_dense_tensor;
    buff->reserve({batch_size_per_gpu * local_gpu_count, label_dim + dense_dim},
                  &temp_dense_tensor);
    current_thread_buffer->device_dense_buffers = temp_dense_tensor.shrink();

    current_thread_buffer->state.store(BufferState::ReadyForWrite);
    current_thread_buffer->current_batch_size = 0;
    current_thread_buffer->batch_size = batchsize;
    current_thread_buffer->param_num = params.size();
    current_thread_buffer->label_dim = label_dim;
    current_thread_buffer->dense_dim = dense_dim;
    current_thread_buffer->batch_size_start_idx =
        batch_size_per_gpu * resource_manager_->get_gpu_global_id_from_local_id(0);
    current_thread_buffer->batch_size_end_idx =
        current_thread_buffer->batch_size_start_idx + batch_size_per_gpu * local_gpu_count;
  }

  broadcast_buffer_->sparse_buffers.reserve(local_gpu_count * params.size());
  broadcast_buffer_->is_fixed_length.reserve(local_gpu_count * params.size());
  broadcast_buffer_->dense_tensors.reserve(local_gpu_count);
  broadcast_buffer_->finish_broadcast_events.resize(local_gpu_count);
  broadcast_buffer_->state.store(BufferState::ReadyForWrite);
  broadcast_buffer_->current_batch_size = 0;
  broadcast_buffer_->param_num = params.size();
  output_->dense_tensors.reserve(local_gpu_count);
  output_->label_tensors.reserve(local_gpu_count);
  output_->use_mixed_precision = use_mixed_precision;
  output_->label_dense_dim = label_dim + dense_dim;
  for (size_t param_id = 0; param_id < params.size(); ++param_id) {
    auto &param = params_[param_id];

    output_->sparse_tensors_map[param.top_name].reserve(local_gpu_count);
    output_->sparse_name_vec.push_back(param.top_name);
  }

  for (size_t local_id = 0; local_id < local_gpu_count; ++local_id) {
    auto local_gpu = resource_manager_->get_local_gpu(local_id);
    CudaDeviceContext ctx(local_gpu->get_device_id());
    auto &buff = buffs[local_id];

    for (size_t param_id = 0; param_id < params.size(); ++param_id) {
      auto &param = params_[param_id];
      SparseTensor<TypeKey> temp_sparse_tensor;
      buff->reserve({(size_t)batchsize, (size_t)param.max_feature_num}, param.slot_num,
                    &temp_sparse_tensor);
      broadcast_buffer_->sparse_buffers.push_back(temp_sparse_tensor.shrink());

      broadcast_buffer_->is_fixed_length.push_back(param.is_fixed_length);
    }
    Tensor2<float> temp_dense_tensor;
    buff->reserve({batch_size_per_gpu, label_dim + dense_dim}, &temp_dense_tensor);
    broadcast_buffer_->dense_tensors.push_back(temp_dense_tensor.shrink());

    HCTR_LIB_THROW(cudaEventCreateWithFlags(&broadcast_buffer_->finish_broadcast_events[local_id],
                                            cudaEventDisableTiming));

    for (size_t param_id = 0; param_id < params.size(); ++param_id) {
      auto &param = params_[param_id];

      SparseTensor<TypeKey> temp_sparse_tensor;
      buff->reserve({(size_t)batchsize, (size_t)param.max_feature_num}, param.slot_num,
                    &temp_sparse_tensor);
      output_->sparse_tensors_map[param.top_name].push_back(temp_sparse_tensor.shrink());
    }

    Tensor2<float> label_tensor;
    buff->reserve({batch_size_per_gpu, label_dim}, &label_tensor);
    output_->label_tensors.push_back(label_tensor.shrink());

    if (use_mixed_precision) {
      Tensor2<__half> dense_tensor;
      buff->reserve({(size_t)batch_size_per_gpu, (size_t)dense_dim}, &dense_tensor);
      output_->dense_tensors.push_back(dense_tensor.shrink());
    } else {
      Tensor2<float> dense_tensor;
      buff->reserve({(size_t)batch_size_per_gpu, (size_t)dense_dim}, &dense_tensor);
      output_->dense_tensors.push_back(dense_tensor.shrink());
    }

    buff->allocate();
  }

  data_collector_ = std::make_shared<DataCollector<TypeKey>>(thread_buffers_, broadcast_buffer_,
                                                             output_, resource_manager);
}
template <typename TypeKey>
DataReader<TypeKey>::~DataReader() {
  try {
    // stop all the loops
    data_collector_->stop();
    if (worker_group_) {
      worker_group_->end();
    }
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    for (size_t i = 0; i < local_gpu_count; ++i) {
      HCTR_LIB_THROW(cudaEventDestroy(broadcast_buffer_->finish_broadcast_events[i]));
    }
  } catch (const std::runtime_error &rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
}
template <typename TypeKey>
long long DataReader<TypeKey>::read_a_batch_to_device() {
  current_batchsize_ = read_a_batch_to_device_delay_release();
  ready_to_collect();
  return current_batchsize_;
}
template <typename TypeKey>
long long DataReader<TypeKey>::read_a_batch_to_device_delay_release() {
  current_batchsize_ = data_collector_->read_a_batch_to_device();
  return current_batchsize_;
}
template <typename TypeKey>
void DataReader<TypeKey>::ready_to_collect() {
  data_collector_->finalize_batch();
}

template <typename TypeKey>
long long DataReader<TypeKey>::get_current_batchsize_per_device(size_t local_id) {
  if (batchsize_ % resource_manager_->get_global_gpu_count() != 0) {
    HCTR_OWN_THROW(Error_t::UnspecificError,
                   "batchsize_ % resource_manager_->get_global_gpu_count() != 0");
  }
  long long batchsize_per_device = batchsize_ / resource_manager_->get_global_gpu_count();
  size_t global_id = resource_manager_->get_gpu_global_id_from_local_id(local_id);
  long long remain_samples = current_batchsize_ - global_id * batchsize_per_device;
  if (remain_samples >= batchsize_per_device) {
    return batchsize_per_device;
  } else if (remain_samples > 0) {
    return remain_samples;
  } else {
    return 0;
  }
}
template <typename TypeKey>
long long DataReader<TypeKey>::get_full_batchsize() const {
  return batchsize_;
}
template <typename TypeKey>
bool DataReader<TypeKey>::current_batch_incomplete() const {
  return current_batchsize_ != batchsize_;
  ;
}
template <typename TypeKey>
bool DataReader<TypeKey>::is_started() const {
  return worker_group_ && worker_group_->is_started();
}
template <typename TypeKey>
void DataReader<TypeKey>::start() {
  if (worker_group_ != nullptr) {
    worker_group_->start();
  } else {
    throw internal_runtime_error(Error_t::NotInitialized, "worker_group_ == nullptr");
  }
}
template <typename TypeKey>
const std::vector<SparseTensorBag> &DataReader<TypeKey>::get_sparse_tensors(
    const std::string &name) {
  if (output_->sparse_tensors_map.find(name) == output_->sparse_tensors_map.end()) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "no such sparse output in data reader:" + name);
  }
  return output_->sparse_tensors_map[name];
}
template <typename TypeKey>
const std::vector<TensorBag2> &DataReader<TypeKey>::get_label_tensors() const {
  return output_->label_tensors;
}
template <typename TypeKey>
const std::vector<TensorBag2> &DataReader<TypeKey>::get_dense_tensors() const {
  return output_->dense_tensors;
}
template <typename TypeKey>
void DataReader<TypeKey>::set_source(std::string file_name) {
  if (worker_group_ != nullptr) {
    if (file_name.empty()) {
      if (file_name_.empty()) {
        throw internal_runtime_error(Error_t::NotInitialized, "invalid file_name");
      } else {
        file_name = file_name_;
      }
    }
    worker_group_->set_source(source_type_, file_name, repeat_, data_source_params_);
  } else {
    throw internal_runtime_error(Error_t::NotInitialized, "worker_group_ == nullptr");
  }
}
template <typename TypeKey>
void DataReader<TypeKey>::create_drwg_norm(std::string file_name, Check_t check_type,
                                           bool start_reading_from_beginning) {
  source_type_ = SourceType_t::FileList;
  worker_group_.reset(
      new DataReaderWorkerGroupNorm<TypeKey>(thread_buffers_, resource_manager_, file_name, repeat_,
                                             check_type, params_, start_reading_from_beginning));
  file_name_ = file_name;
}
template <typename TypeKey>
void DataReader<TypeKey>::create_drwg_raw(std::string file_name, long long num_samples,
                                          bool float_label_dense, bool data_shuffle,
                                          bool start_reading_from_beginning) {
  // check if key type compatible with dataset
  size_t file_size = std::filesystem::file_size(file_name);
  size_t expected_file_size = (label_dim_ + dense_dim_) * sizeof(float);
  for (auto &param : params_) {
    expected_file_size += param.slot_num * sizeof(TypeKey);
  }
  expected_file_size *= num_samples;
  if (file_size != expected_file_size) {
    HCTR_OWN_THROW(Error_t::UnspecificError,
                   "dataset key type and dataset size is not compatible.");
  }
  source_type_ = SourceType_t::Mmap;
  worker_group_.reset(new DataReaderWorkerGroupRaw<TypeKey>(
      thread_buffers_, resource_manager_, file_name, num_samples, repeat_, params_, label_dim_,
      dense_dim_, batchsize_, float_label_dense, data_shuffle, start_reading_from_beginning));
  file_name_ = file_name;
}
#ifndef DISABLE_CUDF
template <typename TypeKey>
void DataReader<TypeKey>::create_drwg_parquet(std::string file_list, bool strict_order_of_batches,
                                              const std::vector<long long> slot_offset,
                                              bool start_reading_from_beginning,
                                              long long max_samples_per_group, int label_dense_num,
                                              int label_dense_dim) {
  source_type_ = SourceType_t::Parquet;
  // worker_group_.empty
  worker_group_.reset(new DataReaderWorkerGroupParquet<TypeKey>(
      thread_buffers_, file_list, strict_order_of_batches, repeat_, params_, slot_offset,
      data_source_params_, resource_manager_, start_reading_from_beginning, label_dense_num,
      label_dense_dim, max_samples_per_group));
}
#endif

template class DataReader<long long int>;
template class DataReader<uint32_t>;
template class core23_reader::DataReader<long long int>;
template class core23_reader::DataReader<uint32_t>;
}  // namespace HugeCTR

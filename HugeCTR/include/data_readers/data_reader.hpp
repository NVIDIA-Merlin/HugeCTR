/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#pragma once

#include <atomic>
#include <common.hpp>
#include <data_reader.hpp>
#include <data_readers/csr.hpp>
#include <data_readers/data_collector.hpp>
#include <data_readers/data_reader_common.hpp>
#include <data_readers/data_reader_worker_group.hpp>
#include <data_readers/data_reader_worker_group_norm.hpp>

#ifndef DISABLE_CUDF
#include <data_readers/data_reader_worker_group_parquet.hpp>
#endif

#include <data_readers/data_reader_worker_group_raw.hpp>
#include <filesystem>
#include <fstream>
#include <gpu_resource.hpp>
#include <tensor2.hpp>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

/**
 * @brief Data reading controller.
 *
 * Control the data reading from data set to embedding.
 * An instance of DataReader will maintain independent
 * threads for data reading (IDataReaderWorker)
 * from dataset to heap. Meanwhile one independent
 * thread consumes the data (DataCollector),
 * and copy the data to GPU buffer.
 */
template <typename TypeKey>
class DataReader : public IDataReader {
 private:
  std::vector<std::shared_ptr<ThreadBuffer>> thread_buffers_;  // gpu_id -> thread_idx
  std::shared_ptr<BroadcastBuffer> broadcast_buffer_;
  std::shared_ptr<DataReaderOutput> output_;

  std::shared_ptr<DataReaderWorkerGroup> worker_group_;
  std::shared_ptr<DataCollector<TypeKey>> data_collector_; /**< pointer of DataCollector */

  /* Each gpu will have several csr output for different embedding */
  const std::vector<DataReaderSparseParam> params_;
  std::shared_ptr<ResourceManager> resource_manager_; /**< gpu resource used in this data reader*/
  const size_t batchsize_;                            /**< batch size */
  const size_t label_dim_; /**< dimention of label e.g. 1 for BinaryCrossEntropy */
  const size_t dense_dim_; /**< dimention of dense */
  long long current_batchsize_;

  bool repeat_;
  std::string file_name_;
  SourceType_t source_type_;

 public:
  DataReader(int batchsize, size_t label_dim, int dense_dim,
             std::vector<DataReaderSparseParam> &params,
             const std::shared_ptr<ResourceManager> &resource_manager, bool repeat, int num_threads,
             bool use_mixed_precision)
      : broadcast_buffer_(new BroadcastBuffer()),
        output_(new DataReaderOutput()),
        params_(params),
        resource_manager_(resource_manager),
        batchsize_(batchsize),
        label_dim_(label_dim),
        dense_dim_(dense_dim),
        repeat_(repeat) {
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
    return;
  }

  ~DataReader() override {
    try {
      // stop all the loops
      data_collector_->stop();
      worker_group_->end();
      size_t local_gpu_count = resource_manager_->get_local_gpu_count();
      for (size_t i = 0; i < local_gpu_count; ++i) {
        HCTR_LIB_THROW(cudaEventDestroy(broadcast_buffer_->finish_broadcast_events[i]));
      }
    } catch (const std::runtime_error &rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    }
  }

  /**
   * Reading a batch from cpu to gpu (embedding)
   */
  TensorScalarType get_scalar_type() const override {
    return TensorScalarTypeFunc<TypeKey>::get_type();
  }

  long long read_a_batch_to_device() override {
    current_batchsize_ = read_a_batch_to_device_delay_release();
    ready_to_collect();
    return current_batchsize_;
  }  // read data from csr to tensors

  long long read_a_batch_to_device_delay_release() override {
    current_batchsize_ = data_collector_->read_a_batch_to_device();
    return current_batchsize_;
  }

  void ready_to_collect() override { data_collector_->finalize_batch(); }

  long long get_current_batchsize_per_device(size_t local_id) override {
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

  long long get_full_batchsize() const override { return batchsize_; }

  bool is_started() const override { return worker_group_->is_started(); }

  void start() override { worker_group_->start(); }

  const std::vector<SparseTensorBag> &get_sparse_tensors(const std::string &name) {
    if (output_->sparse_tensors_map.find(name) == output_->sparse_tensors_map.end()) {
      HCTR_OWN_THROW(Error_t::IllegalCall, "no such sparse output in data reader:" + name);
    }
    return output_->sparse_tensors_map[name];
  }

  const std::vector<TensorBag2> &get_label_tensors() const { return output_->label_tensors; }

  const std::vector<TensorBag2> &get_dense_tensors() const { return output_->dense_tensors; }

  void create_drwg_norm(std::string file_name, Check_t check_type,
                        bool start_reading_from_beginning = true) override {
    source_type_ = SourceType_t::FileList;
    worker_group_.reset(new DataReaderWorkerGroupNorm<TypeKey>(
        thread_buffers_, resource_manager_, file_name, repeat_, check_type, params_,
        start_reading_from_beginning));
    file_name_ = file_name;
  }

  void create_drwg_raw(std::string file_name, long long num_samples, bool float_label_dense,
                       bool data_shuffle = false,
                       bool start_reading_from_beginning = true) override {
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
  void create_drwg_parquet(std::string file_name, const std::vector<long long> slot_offset,
                           bool start_reading_from_beginning = true) override {
    source_type_ = SourceType_t::Parquet;
    // worker_group_.empty
    worker_group_.reset(new DataReaderWorkerGroupParquet<TypeKey>(
        thread_buffers_, file_name, repeat_, params_, slot_offset, resource_manager_,
        start_reading_from_beginning));
  }
#endif

  void set_source(std::string file_name = std::string()) override {
    if (worker_group_ != nullptr) {
      if (file_name.empty()) {
        if (file_name_.empty()) {
          throw internal_runtime_error(Error_t::NotInitialized, "invalid file_name");
        } else {
          file_name = file_name_;
        }
      }
      worker_group_->set_source(source_type_, file_name, repeat_);
    } else {
      throw internal_runtime_error(Error_t::NotInitialized, "worker_group_ == nullptr");
    }
  }
};
}  // namespace HugeCTR

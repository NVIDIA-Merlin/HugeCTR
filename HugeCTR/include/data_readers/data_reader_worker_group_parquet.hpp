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
#pragma once

#include <data_readers/data_reader_worker_group.hpp>
#include <data_readers/dataframe_container.hpp>
#include <data_readers/parquet_data_reader_worker.hpp>

namespace HugeCTR {
template <typename TypeKey>
class DataReaderWorkerGroupParquet : public DataReaderWorkerGroup {
  std::shared_ptr<Source> create_source(size_t worker_id, size_t num_worker,
                                        const std::string& file_name, bool repeat,
                                        const DataSourceParams& data_source_params) override {
    return std::make_shared<ParquetFileSource>(
        worker_id, num_worker, file_name, strict_order_of_batches_, repeat, data_source_params);
  }

 public:
  DataReaderWorkerGroupParquet(const std::vector<std::shared_ptr<ThreadBuffer>>& output_buffers,
                               std::string file_list, bool strict_order_of_batches, bool repeat,
                               const std::vector<DataReaderSparseParam> params,
                               const std::vector<long long> slot_offset,
                               const DataSourceParams data_source_params,
                               const std::shared_ptr<ResourceManager>& resource_manager_,
                               bool start_reading_from_beginning = true, int label_dense_num = 0,
                               int label_dense_dim = 0, long long max_samples_per_group = 0)
      : DataReaderWorkerGroup(start_reading_from_beginning, DataReaderType_t::Parquet,
                              strict_order_of_batches, std::make_shared<std::vector<size_t>>(),
                              output_buffers.size()) {
    if (file_list.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "file_name.empty()");
    }
    // create data reader workers
    size_t num_workers = output_buffers.size();
    auto batchsize = output_buffers[0]->batch_size;
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    std::vector<int> variadic_slots_id;
    std::vector<int> fixed_slot_dims;
    std::vector<size_t> max_sparse_size;
    if (num_workers > local_gpu_count) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "parquet workers should be no greater than local_gpu_count");
    }
    size_t num_slots = 0;
    int global_slot_id = 0;
    for (auto& param : params) {
      int local_slot_id = 0;

      for (const auto& slot_fixed : param.is_slot_fixed_length) {
        max_sparse_size.push_back(param.nnz_per_slot[local_slot_id]);
        if (slot_fixed) {
          fixed_slot_dims.push_back(param.nnz_per_slot[local_slot_id]);
        } else {
          variadic_slots_id.push_back(global_slot_id);
        }
        local_slot_id++;
        global_slot_id++;
      }

      if (param.max_feature_num <= 0 || param.slot_num <= 0) {
        HCTR_OWN_THROW(Error_t::WrongInput, "param.max_feature_num <= 0 || param.slot_num <= 0");
      }
      num_slots += param.slot_num;
    }
    HCTR_CHECK_HINT(num_slots == fixed_slot_dims.size() + variadic_slots_id.size(),
                    " cat columns (fixed + variable) mismatch\n");
    this->set_resource_manager(resource_manager_);
    auto local_device_list = resource_manager_->get_local_gpu_device_id_list();
    size_t dense_bytes_per_sample = sizeof(float) * label_dense_dim;
    // each buffer allocate 2 row_group + batchsize
    long long consumer_rows = max_samples_per_group + batchsize * num_workers;

    std::vector<std::shared_ptr<DFContainer<TypeKey>>> df_container_producer;
    std::vector<std::shared_ptr<DFContainer<TypeKey>>> df_container_consumer;

    // std::vector<size_t>(label_dense_num, 1) implies that all dtypes of dense columns are
    // scalar, it will be initialized until the first iteration begins by the data reader worker
    workers_has_read_.resize(num_workers * num_workers, 0);
    for (size_t i = 0; i < num_workers; i++) {
      df_container_producer_.emplace_back(std::make_shared<DFContainer<TypeKey>>(
          local_device_list[i], max_samples_per_group, std::vector<size_t>(label_dense_num, 0),
          max_sparse_size, dense_bytes_per_sample * max_samples_per_group));

      df_container_producer.push_back(
          std::dynamic_pointer_cast<DFContainer<TypeKey>>(df_container_producer_.back()));
      // for epoch mode , initial state should be EOF
      auto initial_state = BufferState::ReadyForWrite;
      df_container_producer_stats_.emplace_back(
          std::make_shared<std::atomic<BufferState>>(initial_state));
      accomplished_workers_.emplace_back(
          std::make_shared<std::atomic<int>>(strict_order_of_batches_ ? num_workers : 1));
      df_container_consumer_.emplace_back(std::make_shared<DFContainer<TypeKey>>(
          local_device_list[i], consumer_rows, std::vector<size_t>(label_dense_num, 0),
          max_sparse_size, consumer_rows * dense_bytes_per_sample));
      df_container_consumer.push_back(
          std::dynamic_pointer_cast<DFContainer<TypeKey>>(df_container_consumer_.back()));
    }

    for (size_t i = 0; i < num_workers; i++) {
      std::shared_ptr<IDataReaderWorker> data_reader(new ParquetDataReaderWorker<TypeKey>(
          i, num_workers, resource_manager_->get_local_gpu(i % local_gpu_count),
          data_reader_loop_flag_, &this->end_flag_, output_buffers[i], file_list,
          strict_order_of_batches, repeat, params, data_source_params, slot_offset,
          local_device_list[i], df_container_consumer[i], df_container_producer,
          df_container_producer_stats_, workers_has_read_, accomplished_workers_, resource_manager_,
          dense_width_dim_, this->go_next_epoch_.data() + i, this->epoch_mtx_[i],
          this->epoch_cv_[i]));
      data_readers_.push_back(data_reader);
    }
    this->create_data_reader_threads();
  }
  ~DataReaderWorkerGroupParquet() {}
};
}  // namespace HugeCTR

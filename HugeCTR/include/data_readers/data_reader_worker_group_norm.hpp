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

#include <data_readers/data_reader_worker.hpp>
#include <data_readers/data_reader_worker_group.hpp>

namespace HugeCTR {

namespace core23_reader {
template <typename TypeKey>
class DataReaderWorkerGroupNorm : public DataReaderWorkerGroup {
  std::string file_list_; /**< file list of data set */

  std::shared_ptr<Source> create_source(size_t worker_id, size_t num_worker,
                                        const std::string &file_name, bool repeat,
                                        const DataSourceParams &data_source_params) override {
    return std::make_shared<FileSource>(worker_id, num_worker, file_name, repeat);
  }

 public:
  // Ctor
  DataReaderWorkerGroupNorm(const std::vector<std::shared_ptr<ThreadBuffer23>> &output_buffers,
                            const std::shared_ptr<ResourceManager> &resource_manager_,
                            std::string file_list, bool repeat, Check_t check_type,
                            const std::vector<DataReaderSparseParam> &params,
                            bool start_reading_from_beginning = true)
      : DataReaderWorkerGroup(start_reading_from_beginning, DataReaderType_t::Norm, false, nullptr,
                              output_buffers.size()) {
    if (file_list.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "file_name.empty()");
    }
    int num_threads = output_buffers.size();
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();

    // create data reader workers
    int max_feature_num_per_sample = 0;
    for (auto &param : params) {
      max_feature_num_per_sample += param.max_feature_num;

      if (param.max_feature_num <= 0 || param.slot_num <= 0) {
        HCTR_OWN_THROW(Error_t::WrongInput, "param.max_feature_num <= 0 || param.slot_num <= 0");
      }
    }

    set_resource_manager(resource_manager_);
    for (int i = 0; i < num_threads; i++) {
      std::shared_ptr<IDataReaderWorker> data_reader(new core23_reader::DataReaderWorker<TypeKey>(
          i, num_threads, resource_manager_->get_local_gpu(i % local_gpu_count),
          data_reader_loop_flag_, output_buffers[i], file_list, max_feature_num_per_sample, repeat,
          check_type, params));
      data_readers_.push_back(data_reader);
    }
    create_data_reader_threads();
  }
};
}  // namespace core23_reader
template <typename TypeKey>
class DataReaderWorkerGroupNorm : public DataReaderWorkerGroup {
  std::string file_list_; /**< file list of data set */

  std::shared_ptr<Source> create_source(size_t worker_id, size_t num_worker,
                                        const std::string &file_name, bool repeat,
                                        const DataSourceParams &data_source_params) override {
    return std::make_shared<FileSource>(worker_id, num_worker, file_name, repeat);
  }

 public:
  // Ctor
  DataReaderWorkerGroupNorm(const std::vector<std::shared_ptr<ThreadBuffer>> &output_buffers,
                            const std::shared_ptr<ResourceManager> &resource_manager_,
                            std::string file_list, bool repeat, Check_t check_type,
                            const std::vector<DataReaderSparseParam> &params,
                            bool start_reading_from_beginning = true)
      : DataReaderWorkerGroup(start_reading_from_beginning, DataReaderType_t::Norm, false, nullptr,
                              output_buffers.size()) {
    if (file_list.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "file_name.empty()");
    }
    int num_threads = output_buffers.size();
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();

    // create data reader workers
    int max_feature_num_per_sample = 0;
    for (auto &param : params) {
      max_feature_num_per_sample += param.max_feature_num;

      if (param.max_feature_num <= 0 || param.slot_num <= 0) {
        HCTR_OWN_THROW(Error_t::WrongInput, "param.max_feature_num <= 0 || param.slot_num <= 0");
      }
    }

    set_resource_manager(resource_manager_);
    for (int i = 0; i < num_threads; i++) {
      std::shared_ptr<IDataReaderWorker> data_reader(new DataReaderWorker<TypeKey>(
          i, num_threads, resource_manager_->get_local_gpu(i % local_gpu_count),
          data_reader_loop_flag_, output_buffers[i], file_list, max_feature_num_per_sample, repeat,
          check_type, params));
      data_readers_.push_back(data_reader);
    }
    create_data_reader_threads();
  }
};
}  // namespace HugeCTR

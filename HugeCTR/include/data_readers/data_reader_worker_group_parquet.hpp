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

#include <data_readers/data_reader_worker_group.hpp>
#include <data_readers/parquet_data_reader_worker.hpp>

namespace HugeCTR {

template <typename TypeKey>
class DataReaderWorkerGroupParquet : public DataReaderWorkerGroup {
  std::shared_ptr<Source> create_source(size_t worker_id, size_t num_worker,
      const std::string& file_name, bool repeat) override {
    CK_THROW_(Error_t::UnspecificError, "Invalid Call");
    return std::shared_ptr<Source>(reinterpret_cast<Source*>(new int));
  }

 public:
  // Ctor
  DataReaderWorkerGroupParquet(std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> csr_heap,
                               std::string file_list,
                               const std::vector<DataReaderSparseParam> params,
                               const std::vector<long long> slot_offset,
                               const std::shared_ptr<ResourceManager> resource_manager,
                               bool start_reading_from_beginning = true)
      : DataReaderWorkerGroup(start_reading_from_beginning, DataReaderType_t::Parquet) {
    if (file_list.empty()) {
      CK_THROW_(Error_t::WrongInput, "file_name.empty()");
    }
    // create data reader workers
    int max_feature_num_per_sample = 0;
    for (auto& param : params) {
      max_feature_num_per_sample += param.max_feature_num;

      if (param.max_feature_num <= 0 || param.slot_num <= 0) {
        CK_THROW_(Error_t::WrongInput, "param.max_feature_num <= 0 || param.slot_num <= 0");
      }
    }

    this->set_resource_manager(resource_manager);
    auto local_device_list = resource_manager_->get_local_gpu_device_id_list();
    int NumThreads = csr_heap->get_size();
    for (int i = 0; i < NumThreads; i++) {
      std::shared_ptr<IDataReaderWorker> data_reader(new ParquetDataReaderWorker<TypeKey>(
          i, NumThreads, csr_heap, file_list, max_feature_num_per_sample, params, slot_offset,
          local_device_list[i], resource_manager_));
      data_readers_.push_back(data_reader);
    }
    create_data_reader_threads();
  }
};
}  // namespace HugeCTR

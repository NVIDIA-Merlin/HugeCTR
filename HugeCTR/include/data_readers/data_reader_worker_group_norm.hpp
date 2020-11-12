/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

template <typename TypeKey>
class DataReaderWorkerGroupNorm : public DataReaderWorkerGroup {
  std::string file_list_; /**< file list of data set */
 public:
  // Ctor
  DataReaderWorkerGroupNorm(std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> csr_heap,
                            std::string file_list,
                            bool repeat,
                            Check_t check_type,
                            const std::vector<DataReaderSparseParam> params,
                            bool start_reading_from_beginning = true)
      : DataReaderWorkerGroup(start_reading_from_beginning) {
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
    int NumThreads = csr_heap->get_size();
    for (int i = 0; i < NumThreads; i++) {
      std::shared_ptr<IDataReaderWorker> data_reader(new DataReaderWorker<TypeKey>(
          i, NumThreads, csr_heap, file_list, max_feature_num_per_sample, repeat, check_type, params));
      data_readers_.push_back(data_reader);
    }
    create_data_reader_threads();
  }
};
}  // namespace HugeCTR

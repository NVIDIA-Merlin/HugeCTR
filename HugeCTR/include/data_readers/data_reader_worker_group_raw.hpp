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

#include <data_readers/data_reader_worker_group.hpp>
#include <data_readers/data_reader_worker_raw.hpp>

namespace HugeCTR {

template <typename TypeKey>
class DataReaderWorkerGroupRaw : public DataReaderWorkerGroup{
  std::shared_ptr<MmapOffsetList> file_offset_list_;
  
public:
  //Ctor
  DataReaderWorkerGroupRaw(std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> csr_heap, std::string file_name, long long num_samples, const std::vector<DataReaderSparseParam> params, const std::vector<long long> slot_offset, int label_dim, int dense_dim, int batchsize, bool data_shuffle = false, bool start_reading_from_beginning = true): DataReaderWorkerGroup(start_reading_from_beginning) {
    //todo param check
    if(file_name.empty()){
      CK_THROW_(Error_t::WrongInput, "file_name.empty()");
    }

    {
      int slots = 0;
      for (auto& param : params) {
	slots += param.slot_num;
      }
      file_offset_list_.reset(new MmapOffsetList(
	 file_name, num_samples, (label_dim + dense_dim + slots) * sizeof(int), batchsize,
         data_shuffle, csr_heap->get_size()));
    }

    for (int i = 0; i < csr_heap->get_size(); i++) {
      std::shared_ptr<IDataReaderWorker> data_reader(
	new DataReaderWorkerRaw<TypeKey>(i, csr_heap->get_size(), file_offset_list_, csr_heap,
        file_name, params, slot_offset, label_dim));
      data_readers_.push_back(data_reader);
    }
    create_data_reader_threads();
  }
};
}// namespace HugeCTR 

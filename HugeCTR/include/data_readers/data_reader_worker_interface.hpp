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
#include <common.hpp>
#include <data_readers/source.hpp>
#include <memory>

namespace HugeCTR {
class IDataReaderWorker {
 public:
  virtual void read_a_batch() = 0;
  virtual void skip_read() = 0;
  void set_source(std::shared_ptr<Source> source) {
    if (!is_eof_) {
      CK_THROW_(Error_t::IllegalCall,
                "DataSource cannot be changed in the \"repeat\" mode or when a data reader worker "
                "is not in the EOF state.");
    }

    pre_set_source();
    source_ = source;
    post_set_source();
  }

  IDataReaderWorker() : is_eof_(false) {}

 protected:
  std::shared_ptr<Source> source_; /**< source: can be file or network */
  bool is_eof_;

 private:
  virtual void pre_set_source() {}
  virtual void post_set_source() {}
};

template <typename T>
void fill_empty_sample(std::vector<DataReaderSparseParam>& params, CSRChunk<T>* csr_chunk) {
  int param_id = 0;
  for (auto& param : params) {
    for (int k = 0; k < param.slot_num; k++) {
      if (param.type == DataReaderSparse_t::Distributed) {
        for (int dev_id = 0; dev_id < csr_chunk->get_num_devices(); dev_id++) {
          csr_chunk->get_csr_buffer(param_id, dev_id).new_row();
        }
      } else if (param.type == DataReaderSparse_t::Localized) {
        int dev_id = k % csr_chunk->get_num_devices();
        csr_chunk->get_csr_buffer(param_id, dev_id).new_row();
      } else {
        CK_THROW_(Error_t::UnspecificError, "param.type is not defined");
      }
    }
    param_id++;
  }  // for(auto& param: params)
}

}  // namespace HugeCTR

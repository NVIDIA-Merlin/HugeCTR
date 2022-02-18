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
#include <data_readers/data_reader_common.hpp>
#include <data_readers/source.hpp>
#include <memory>

namespace HugeCTR {

class IDataReaderWorker {
 public:
  virtual void read_a_batch(){};
  virtual void skip_read(){};
  void set_source(std::shared_ptr<Source> source) {
    if (!is_eof_) {
      HCTR_OWN_THROW(
          Error_t::IllegalCall,
          "DataSource cannot be changed in the \"repeat\" mode or when a data reader worker "
          "is not in the EOF state.");
    }
    pre_set_source();
    source_ = source;
    post_set_source();
  }
  ~IDataReaderWorker(){};
  IDataReaderWorker() : is_eof_(false) {}

 protected:
  std::shared_ptr<Source> source_; /**< source: can be file or network */

  int worker_id_;
  int worker_num_;
  std::shared_ptr<GPUResource> gpu_resource_;

  bool is_eof_;
  int *loop_flag_;

  std::shared_ptr<ThreadBuffer> buffer_;

  IDataReaderWorker(const int worker_id, const int worker_num,
                    const std::shared_ptr<GPUResource> &gpu_resource, bool is_eof, int *loop_flag,
                    const std::shared_ptr<ThreadBuffer> &buff)
      : worker_id_(worker_id),
        worker_num_(worker_num),
        gpu_resource_(gpu_resource),
        is_eof_(is_eof),
        loop_flag_(loop_flag),
        buffer_(buff) {}

  bool wait_until_h2d_ready() {
    BufferState expected = BufferState::ReadyForWrite;
    while (!buffer_->state.compare_exchange_weak(expected, BufferState::Writing)) {
      expected = BufferState::ReadyForWrite;
      usleep(2);
      if (*loop_flag_ == 0) return false;  // in case main thread exit
    }
    return true;
  }

 private:
  virtual void pre_set_source() {}
  virtual void post_set_source() {}
};

}  // namespace HugeCTR

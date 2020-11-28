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

#include <atomic>
#include <common.hpp>
#include <data_readers/csr.hpp>
#include <data_readers/csr_chunk.hpp>
#include <data_readers/data_reader_worker_interface.hpp>
#include <fstream>
#include <thread>
#include <vector>

namespace HugeCTR {

/**
 * A helper function to read data from dataset to heap in a new thread.
 * @param data_reader a pointer of data_reader.
 * @param p_loop_flag a flag to control the loop,
          and break loop when IDataReaderWorker is destroyed.
 */

static void data_reader_thread_func_(const std::shared_ptr<IDataReaderWorker>& data_reader,
                                     int* p_loop_flag) {
  try {
    while ((*p_loop_flag) == 0) {
      usleep(2);
    }

    while (*p_loop_flag) {
      data_reader->read_a_batch();
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

class DataReaderWorkerGroup {
  std::vector<std::thread> data_reader_threads_; /**< A vector of the pointers of data reader .*/
  int data_reader_loop_flag_{0};                 /**< p_loop_flag a flag to control the loop */
 protected:
  std::vector<std::shared_ptr<IDataReaderWorker>>
      data_readers_; /**< A vector of DataReaderWorker' pointer.*/
  std::shared_ptr<ResourceManager> resource_manager_;
  /**
   * Create threads to run data reader workers
   */
  void create_data_reader_threads() {
    if (data_readers_.empty()) {
      CK_THROW_(Error_t::WrongInput, "data_readers_.empty()");
    }
    if (!data_reader_threads_.empty()) {
      CK_THROW_(Error_t::WrongInput, "!data_reader_threads_.empty()");
    }

    for (auto& data_reader : data_readers_) {
      data_reader_threads_.emplace_back(data_reader_thread_func_, data_reader,
                                        &data_reader_loop_flag_);
      set_affinity(data_reader_threads_.back(), {}, true);
    }
  }

 public:
  DataReaderWorkerGroup(bool start_reading_from_beginning) {
    if (start_reading_from_beginning) {
      data_reader_loop_flag_ = 1;
    }
  }
  void set_resource_manager(const std::shared_ptr<ResourceManager>& resource_manager) {
    resource_manager_ = resource_manager;
  }
  bool is_started() const { return data_reader_loop_flag_; }
  void start() { data_reader_loop_flag_ = 1; }
  void end() {
    data_reader_loop_flag_ = 0;
    for (auto& data_reader : data_readers_) {
      data_reader->skip_read();
    }
  }
  virtual ~DataReaderWorkerGroup() {
    for (auto& data_reader_thread : data_reader_threads_) {
      data_reader_thread.join();
    }
  }
  template <typename Op>
  void set_source(Op op) {
    size_t num_workers = data_readers_.size();
    for (size_t worker_id = 0; worker_id < num_workers; worker_id++) {
      data_readers_[worker_id]->set_source(op(worker_id, num_workers));
    }
    if (data_reader_loop_flag_ == 0) {
      start();
    }
  }
};
}  // namespace HugeCTR

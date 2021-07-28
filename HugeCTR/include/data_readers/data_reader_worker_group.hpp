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

#include <numa.h>

#include <atomic>
#include <common.hpp>
#include <data_readers/csr.hpp>
#include <data_readers/data_reader_worker_interface.hpp>
#include <fstream>
#include <thread>
#include <tuple>
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
 protected:
  int data_reader_loop_flag_{0}; /**< p_loop_flag a flag to control the loop */
  DataReaderType_t data_reader_type_;
  std::vector<std::shared_ptr<IDataReaderWorker>>
      data_readers_; /**< A vector of DataReaderWorker' pointer.*/
  std::shared_ptr<ResourceManager> resource_manager_;

  std::vector<cpu_set_t> vec_cpu_set_;

  void generate_thread_core_affinity(int num_reader_threads) {
    if (numa_available() < 0) {
      // set to use all cores
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      for (int core = 0; core < 256; core++) CPU_SET(core, &cpuset);
      vec_cpu_set_.push_back(cpuset);
    } else {
      constexpr int thread_per_cpu_node = 8;
      int max_cpu_nodes_required =
          (num_reader_threads + (thread_per_cpu_node - 1)) / thread_per_cpu_node;

      int cpus = 0;
      std::vector<std::vector<int>> cpu_numa_core_arr;
      for (int node = 0; node <= numa_max_node(); node++) {
        if (numa_bitmask_isbitset(numa_nodes_ptr, node)) {
          std::vector<int> core_arr;
          struct bitmask* cpu_mask;
          cpu_mask = numa_allocate_cpumask();
          numa_node_to_cpus(node, cpu_mask);
          int skip_count = 0;
          for (long unsigned int core = 0; core < cpu_mask->size; core++) {
            if (numa_bitmask_isbitset(cpu_mask, core)) {
              if (skip_count < 2) {
                skip_count++;
                continue;
              }
              core_arr.push_back(core);
            }
          }
          cpu_numa_core_arr.push_back(core_arr);
          numa_bitmask_free(cpu_mask);
        }
        cpus++;
        if (cpus == max_cpu_nodes_required) break;
      }

      for (auto core_set : cpu_numa_core_arr) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (auto core : core_set) CPU_SET(core, &cpuset);
        vec_cpu_set_.push_back(cpuset);
      }
    }
  }

  /**
   * Create threads to run data reader workers
>>>>>>> v3.1_preview
   */
  void create_data_reader_threads() {
    if (data_readers_.empty()) {
      CK_THROW_(Error_t::WrongInput, "data_readers_.empty()");
    }
    if (!data_reader_threads_.empty()) {
      CK_THROW_(Error_t::WrongInput, "!data_reader_threads_.empty()");
    }

    // decide thread-core affinity
    generate_thread_core_affinity(data_readers_.size());
    uint32_t tid = 0;
    for (auto& data_reader : data_readers_) {
      data_reader_threads_.emplace_back(data_reader_thread_func_, data_reader,
                                        &data_reader_loop_flag_);
      // set_affinity(data_reader_threads_.back(), {}, true);
      int rc =
          pthread_setaffinity_np(data_reader_threads_.back().native_handle(), sizeof(cpu_set_t),
                                 &(vec_cpu_set_[tid % vec_cpu_set_.size()]));
      if (rc != 0) {
        CK_THROW_(Error_t::WrongInput,
                  "Error calling pthread_setaffinity_np: " + std::to_string(rc));
      }
      tid++;
    }
  }

 public:
  DataReaderWorkerGroup(bool start_reading_from_beginning, DataReaderType_t data_reader_type)
      : data_reader_type_(data_reader_type) {
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
    for (auto& data_reader : data_readers_) {
      data_reader->skip_read();
    }
    // Make sure the data reader  threads escape the pre-main loop
    if (data_reader_loop_flag_ == 0) {
      data_reader_loop_flag_ = 1;
      sleep(2);
    }
    // Data reader threads escape the main loop
    data_reader_loop_flag_ = 0;
  }

  virtual ~DataReaderWorkerGroup() {
    for (auto& data_reader_thread : data_reader_threads_) {
      data_reader_thread.join();
    }
  }

  void set_source(SourceType_t source_type, const std::string& file_name, bool repeat) {
    if (!((source_type == SourceType_t::FileList && data_reader_type_ == DataReaderType_t::Norm) ||
          (source_type == SourceType_t::Mmap && data_reader_type_ == DataReaderType_t::Raw) ||
          (source_type == SourceType_t::Parquet &&
           data_reader_type_ == DataReaderType_t::Parquet))) {
      CK_THROW_(Error_t::WrongInput,
                "set_source only supports FileList for Norm & Mmap for Raw & Parquet for Parquet");
    }
    size_t num_workers = data_readers_.size();
    for (size_t worker_id = 0; worker_id < num_workers; worker_id++) {
      data_readers_[worker_id]->set_source(
          create_source(worker_id, num_workers, file_name, repeat));
    }
    if (data_reader_loop_flag_ == 0) {
      start();
    }
  }

 private:
  virtual std::shared_ptr<Source> create_source(size_t worker_id, size_t num_worker,
                                                const std::string& file_name, bool repeat) = 0;
};
}  // namespace HugeCTR

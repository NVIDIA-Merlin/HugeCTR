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

#include <numa.h>

#include <atomic>
#include <common.hpp>
#include <condition_variable>
#include <data_readers/csr.hpp>
#include <data_readers/data_container_interface.hpp>
#include <data_readers/data_reader_worker_interface.hpp>
#include <fstream>
#include <mutex>
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

static void producer_thread_func_(const std::shared_ptr<IDataReaderWorker>& data_reader,
                                  const std::shared_ptr<std::atomic<bool>>& p_loop_flag,
                                  int device_id, volatile bool* end_flag = nullptr) {
  try {
    // this thread needs numa bind for higher IO bandwidth
    CudaCPUDeviceContext context(device_id);

    while (!p_loop_flag->load()) {
      usleep(2);
    }
    // parquet reader will hangs over here untill end_flag is set true
    // others will return right away
    data_reader->do_h2d();
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
}

static void consumer_thread_func_(const std::shared_ptr<IDataReaderWorker>& data_reader,
                                  const std::shared_ptr<std::atomic<bool>>& p_loop_flag,
                                  int device_id, volatile bool* end_flag = nullptr) {
  try {
    CudaDeviceContext context(device_id);
    while (!p_loop_flag->load() && !*end_flag) {
      usleep(2);
    }
    while (p_loop_flag->load() && !*end_flag) {
      data_reader->read_a_batch();
    }
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
}

class DataReaderWorkerGroup {
  std::vector<std::thread> consumer_threads_; /**< A vector of the pointers of data reader .*/
  std::vector<std::thread> producer_threads_; /**< A vector of the pointers of data reader .*/
 protected:
  // below are init in child class
  // current only parquet supports those parameters
  // in the future, DFContainer will be replaced by more generic container
  // dense_width_dim_ may be removed in the future
  std::vector<std::shared_ptr<UnifiedContainer>> df_container_producer_;
  std::vector<std::shared_ptr<UnifiedContainer>> df_container_consumer_;
  std::vector<std::shared_ptr<std::atomic<BufferState>>> df_container_producer_stats_;
  std::vector<std::shared_ptr<std::atomic<int>>> accomplished_workers_;
  std::vector<char> workers_has_read_;

  volatile bool end_flag_{false};
  std::shared_ptr<std::atomic<bool>>
      data_reader_loop_flag_; /**< p_loop_flag a flag to control the loop */
  // only parquet uses them by far
  std::vector<char> go_next_epoch_;
  std::vector<std::mutex> epoch_mtx_;
  std::vector<std::condition_variable> epoch_cv_;

  DataReaderType_t data_reader_type_;
  std::vector<std::shared_ptr<IDataReaderWorker>>
      data_readers_; /**< A vector of DataReaderWorker' pointer.*/
  std::shared_ptr<ResourceManager> resource_manager_;
  bool strict_order_of_batches_;
  std::shared_ptr<std::vector<size_t>> dense_width_dim_;

  /**
   * Create threads to run data reader workers
   */
  void create_data_reader_threads() {
    if (data_readers_.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "data_readers_.empty()");
    }
    if (!consumer_threads_.empty() || !producer_threads_.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "!consumer_threads_ or producer_threads_ is not empty()");
    }

    size_t local_gpu_count = resource_manager_->get_local_gpu_count();

    for (size_t i = 0; i < data_readers_.size(); ++i) {
      auto local_gpu = resource_manager_->get_local_gpu(i % local_gpu_count);

      consumer_threads_.emplace_back(consumer_thread_func_, data_readers_[i],
                                     data_reader_loop_flag_, local_gpu->get_device_id(),
                                     &this->end_flag_);
      producer_threads_.emplace_back(producer_thread_func_, data_readers_[i],
                                     data_reader_loop_flag_, local_gpu->get_device_id(),
                                     &this->end_flag_);
    }
  }

 public:
  DataReaderWorkerGroup(bool start_reading_from_beginning, DataReaderType_t data_reader_type,
                        bool strict_order_of_batches = false,
                        std::shared_ptr<std::vector<size_t>> dense_width_dim = nullptr,
                        int num_workers = 0)
      : end_flag_(false),
        data_reader_type_(data_reader_type),
        strict_order_of_batches_(strict_order_of_batches),
        dense_width_dim_(dense_width_dim) {
    if (start_reading_from_beginning) {
      data_reader_loop_flag_ = std::make_shared<std::atomic<bool>>(true);
      go_next_epoch_ = std::vector<char>(num_workers, 1);
    } else {
      data_reader_loop_flag_ = std::make_shared<std::atomic<bool>>(false);
      go_next_epoch_ = std::vector<char>(num_workers, 0);
    }
    epoch_mtx_ = std::vector<std::mutex>(num_workers);
    epoch_cv_ = std::vector<std::condition_variable>(num_workers);
  }
  void set_resource_manager(const std::shared_ptr<ResourceManager>& resource_manager) {
    resource_manager_ = resource_manager;
  }
  bool is_started() const { return data_reader_loop_flag_->load(); }
  void start() { data_reader_loop_flag_->store(1); }

  void end() {
    end_flag_ = true;

    for (auto& data_reader : data_readers_) {
      data_reader->skip_read();
    }
    if (!data_reader_loop_flag_->load()) {
      data_reader_loop_flag_->store(1);
      usleep(100);
    }
    for (size_t i = 0; i < data_readers_.size(); ++i) {
      std::unique_lock<std::mutex> lck(this->epoch_mtx_[i]);
      // awken
      go_next_epoch_[i] = (1);
      lck.unlock();
      this->epoch_cv_[i].notify_all();
    }
  }

  virtual ~DataReaderWorkerGroup() {
    data_reader_loop_flag_->store(0);
    for (auto& data_reader_thread : consumer_threads_) {
      data_reader_thread.join();
    }
    for (auto& data_reader_thread : producer_threads_) {
      data_reader_thread.join();
    }
  }
  virtual void pre_set_source(){};
  virtual void post_set_source(){};
  virtual void set_source(SourceType_t source_type, const std::string& file_name, bool repeat,
                          const DataSourceParams& data_source_params,
                          bool strict_order_of_batches = false) {
    if (!((source_type == SourceType_t::FileList && data_reader_type_ == DataReaderType_t::Norm) ||
          (source_type == SourceType_t::Mmap && data_reader_type_ == DataReaderType_t::Raw) ||
          (source_type == SourceType_t::Parquet &&
           data_reader_type_ == DataReaderType_t::Parquet))) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "set_source only supports FileList for Norm & Mmap for Raw & Parquet for Parquet");
    }

    size_t num_workers = data_readers_.size();
    for (size_t worker_id = 0; worker_id < num_workers; worker_id++) {
      data_readers_[worker_id]->pre_set_source();
    }
    for (size_t worker_id = 0; worker_id < num_workers; worker_id++) {
      data_readers_[worker_id]->set_source(
          create_source(worker_id, num_workers, file_name, repeat, data_source_params));
    }
    // Has no impact if data_reader_loop_flag_->load() == false
    for (size_t i = 0; i < num_workers; ++i) {
      std::unique_lock<std::mutex> lck(this->epoch_mtx_[i]);
      // awken
      go_next_epoch_[i] = 1;
      lck.unlock();
      this->epoch_cv_[i].notify_all();
    }
    for (size_t worker_id = 0; worker_id < num_workers; worker_id++) {
      data_readers_[worker_id]->post_set_source();
    }
    if (!data_reader_loop_flag_->load()) {
      start();
    }
  }

 private:
  virtual std::shared_ptr<Source> create_source(size_t worker_id, size_t num_worker,
                                                const std::string& file_name, bool repeat,
                                                const DataSourceParams& data_source_params) = 0;
};
}  // namespace HugeCTR

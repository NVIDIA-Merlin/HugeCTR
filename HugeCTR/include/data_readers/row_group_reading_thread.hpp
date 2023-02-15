/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <condition_variable>
#include <data_readers/dataframe_container.hpp>
#include <data_readers/file_source_parquet.hpp>
#include <resource_managers/resource_manager_ext.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace HugeCTR {

template <typename T>
class RowGroupReadingThread {
  int device_id_;
  int worker_id_;
  int num_workers_;
  int MAX_TRY{10};
  // pausing signal
  // how many workers have copied from df_container_producer_
  const int num_workers_consume_;
  std::vector<std::shared_ptr<std::atomic<int>>>& accomplished_workers_;
  std::vector<char>& workers_has_read_;
  volatile bool* end_loop_;
  std::vector<std::shared_ptr<std::atomic<BufferState>>>& producer_buffer_stats_;
  std::shared_ptr<DFContainer<T>> df_container_consumer_;
  std::vector<std::shared_ptr<DFContainer<T>>> df_container_producer_;
  // interact with filesourceparquet
  ParquetFileSource* source_;
  rmm::mr::device_memory_resource* memory_resource_;
  bool strict_order_of_batches_;
  long long local_row_group_id_;  // reset for new file

  std::unique_ptr<cudf::table> cached_df_;

  std::map<int, int>& dense_idx_to_parquet_col_;
  std::map<int, int>& categorical_idx_parquet_col_;

 public:
  void read_new_file(long long expected_row_groups = 1);
  RowGroupReadingThread(
      int device_id, int worker_id, int num_workers, int consume_workers, volatile bool* end_flag,
      ParquetFileSource* source, rmm::mr::device_memory_resource* memory_resource,
      bool strict_order_of_batches, std::map<int, int>& dense_idx_to_parquet_col,
      std::map<int, int>& categorical_idx_parquet_col,
      std::shared_ptr<DFContainer<T>> df_container_consumer,
      std::vector<std::shared_ptr<DFContainer<T>>>& df_container_producer,
      std::vector<std::shared_ptr<std::atomic<BufferState>>>& producer_buffer_stats,
      std::vector<char>& workers_has_read,
      std::vector<std::shared_ptr<std::atomic<int>>>& accomplished_workers);

  ~RowGroupReadingThread();
  // wait untill num_workers_consume_ workers has copied from df_container_producer_[worker_id_]
  Error_t get_one_read_group(const std::vector<DataReaderSparseParam>& params,
                             std::vector<size_t>& dense_dim_array,
                             std::vector<int>& one_hot_slot_id, std::vector<int>& sparse_nnz_array);

  bool wait_until_writeable(bool bypass = false);
  bool wait_until_readable(int worker_id);
  void inc_accomplished_worker(int worker_id);
  void reset_accomplished_worker();
  void reset_read_flag();
  void set_this_producer_status(BufferState stat);

  void start();
  void stop();

  bool source_available();
  // reset source and then awake producer
  void reset_source(ParquetFileSource* source);
  // only has effect in epoch mode
  bool is_eof(int producer_id_);

  long long get_current_num_row_groups();
  long long get_local_row_group_id();

  std::shared_ptr<DFContainer<T>>& get_df_container_consumer();
  std::shared_ptr<DFContainer<T>>& get_df_container_producer(int worker_id);
  int get_accomplished_workers(int id) { return accomplished_workers_[id]->load(); };
};

}  // namespace HugeCTR
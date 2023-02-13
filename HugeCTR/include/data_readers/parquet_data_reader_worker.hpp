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
#include <common.hpp>
#include <data_readers/data_reader_worker_interface.hpp>
#include <data_readers/dataframe_container.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <data_readers/file_source_parquet.hpp>
#include <data_readers/row_group_reading_thread.hpp>
#include <resource_managers/resource_manager_ext.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#include <condition_variable>
#include <data_readers/file_list.hpp>
#include <data_readers/metadata.hpp>
#include <data_readers/parquet_data_converter.hpp>
#include <memory>
#include <mutex>
#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

template <class T>
class ParquetDataReaderWorker : public IDataReaderWorker,
                                public std::enable_shared_from_this<ParquetDataReaderWorker<T>> {
 private:
  std::vector<DataReaderSparseParam> params_; /**< configuration of data reader sparse input */
  bool skip_read_{false};               /**< set to true when you want to stop the data reading */
  bool strict_order_of_batches_{false}; /**< set to true when you want to sequentially read file*/
  bool repeat_;                         /**< set to true when you want to sequentially read file*/
  const int MAX_TRY = 10;
  int slots_{0};
  std::vector<long long> slot_offset_;
  std::vector<T> slot_offset_dtype_;
  std::shared_ptr<rmm::device_buffer> slot_offset_device_buf_; /**< GPU buffer w/ slot offset*/
  std::shared_ptr<rmm::mr::device_memory_resource>
      memory_resource_;                            /**< RMM device memory resource object */
  int device_id_{0};                               /**< GPU id for execution */
  cudaStream_t task_stream_;                       /**< Stream for Parquet to csr work  */
  cudaStream_t dense_stream_;                      /**< Stream for Parquet to dense work  */
  std::map<int, int> dense_idx_to_parquet_col_;    /**< dense feature to parquet column idx */
  std::map<int, int> categorical_idx_parquet_col_; /**< cat(slot) to parquet column idx */
  bool thread_resource_allocated_; /**< Flag to set/allocate worker thread resources */
  Tensor2<int64_t> host_memory_pointer_staging_;
  // read from parquet table
  std::vector<std::string> column_names_{};

  // temp tensor for dense_dim_array
  Tensor2<int64_t> host_memory_dense_dim_array_;
  Tensor2<int64_t> device_memory_dense_dim_array_;

  /**< Pinned memory for async column dev ptr copy */
  Tensor2<int64_t> host_pinned_csr_inc_;

  long long view_offset_;  // set this to discard row slices in current cached_df_ you want
  std::shared_ptr<ResourceManager> resource_manager_;

  std::shared_ptr<std::vector<size_t>> dense_width_dim_;
  std::vector<int> one_hot_cols_;
  std::vector<int> sparse_nnz_array_;

  int num_label_dense_;
  long long global_row_group_id_;

  ParquetFileSource* parquet_file_source() const {
    return static_cast<ParquetFileSource*>(source_.get());
  }

  std::shared_ptr<RowGroupReadingThread<T>> row_group_reader_;

  char* go_next_epoch_;
  std::mutex& epoch_mtx_;
  std::condition_variable& epoch_cv_;

 public:
  void set_source(std::shared_ptr<Source> source) override {
    if (!source) {
      HCTR_LOG(INFO, WORLD, "source is empty!!\n");
    } else {
      if (source->is_open()) {
        HCTR_LOG(INFO, WORLD, "source is open!!\n");
      }
    }
    this->source_ = source;
    auto& consumer = row_group_reader_->get_df_container_consumer();
    auto& producer = row_group_reader_->get_df_container_producer(worker_id_);

    if (consumer->dense_dim_array_init_) {
      consumer->reset_ptrs();
    }
    if (producer->dense_dim_array_init_) {
      producer->reset_ptrs();
    }
    if (this->row_group_reader_) {
      row_group_reader_->reset_source(static_cast<ParquetFileSource*>(source_.get()));
    }
  }
  void post_set_source() override {
    is_eof_ = false;

    buffer_->state.store(BufferState::ReadyForWrite);
    global_row_group_id_ = 0;
  }
  /**
   * Ctor
   */
  ParquetDataReaderWorker(
      unsigned int worker_id, unsigned int worker_num,
      const std::shared_ptr<GPUResource>& gpu_resource,
      const std::shared_ptr<std::atomic<bool>>& loop_flag, volatile bool* end_flag,
      const std::shared_ptr<ThreadBuffer>& buffer, const std::string& file_list,
      bool strict_order_of_batches, bool repeat, const std::vector<DataReaderSparseParam>& params,
      const DataSourceParams& data_source_params, const std::vector<long long>& slot_offset,
      int device_id, std::shared_ptr<DFContainer<T>> df_container_consumer,
      std::vector<std::shared_ptr<DFContainer<T>>>& df_container_producer,
      std::vector<std::shared_ptr<std::atomic<BufferState>>>& producer_buffer_stats,
      std::vector<char>& workers_has_read,
      std::vector<std::shared_ptr<std::atomic<int>>>& accomplished_workers,
      const std::shared_ptr<ResourceManager>& resource_manager,
      std::shared_ptr<std::vector<size_t>> dense_width_dim_, char* go_next_epoch_,
      std::mutex& epoch_mtx_, std::condition_variable& epoch_cv_);

  ~ParquetDataReaderWorker();
  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch() override;
  void do_h2d() override;
  /**
   * skip data reading in read_a_batch()
   */
  void skip_read() { skip_read_ = true; }
  DataReaderType_t get_reader_type() override { return DataReaderType_t::Parquet; }
};
}  // namespace HugeCTR

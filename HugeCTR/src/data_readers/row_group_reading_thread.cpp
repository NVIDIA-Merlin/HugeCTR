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

#include <data_readers/row_group_reading_thread.hpp>
#include <map>
#include <string>
namespace HugeCTR {

// static std::map<BufferState, std::string> stat_str;
template <typename T>
RowGroupReadingThread<T>::RowGroupReadingThread(
    int device_id, int worker_id, int num_workers, int consume_workers, volatile bool* end_flag,
    ParquetFileSource* source, rmm::mr::device_memory_resource* memory_resource,
    bool strict_order_of_batches, std::map<int, int>& dense_idx_to_parquet_col,
    std::map<int, int>& categorical_idx_parquet_col,
    std::shared_ptr<DFContainer<T>> df_container_consumer,
    std::vector<std::shared_ptr<DFContainer<T>>>& df_container_producer,
    std::vector<std::shared_ptr<std::atomic<BufferState>>>& producer_buffer_stats,
    std::vector<char>& workers_has_read,
    std::vector<std::shared_ptr<std::atomic<int>>>& accomplished_workers)
    : device_id_(device_id),
      worker_id_(worker_id),
      num_workers_(num_workers),
      num_workers_consume_(consume_workers),
      accomplished_workers_(accomplished_workers),
      workers_has_read_(workers_has_read),
      end_loop_(end_flag),
      producer_buffer_stats_(producer_buffer_stats),
      df_container_consumer_(df_container_consumer),
      df_container_producer_(df_container_producer),
      source_(source),
      memory_resource_(memory_resource),
      strict_order_of_batches_(strict_order_of_batches),
      local_row_group_id_(0),
      dense_idx_to_parquet_col_(dense_idx_to_parquet_col),
      categorical_idx_parquet_col_(categorical_idx_parquet_col){};

// one of the consumer will notify producer that it has got the group
template <typename T>
void RowGroupReadingThread<T>::inc_accomplished_worker(int creditor_id) {
  accomplished_workers_[creditor_id]->fetch_add(1);
  this->workers_has_read_[worker_id_ * num_workers_ + creditor_id] = true;
  // this is important!
  if (accomplished_workers_[creditor_id]->load() == num_workers_consume_) {
    this->producer_buffer_stats_[creditor_id]->store(BufferState::ReadyForWrite);
  }
  if (accomplished_workers_[creditor_id]->load() > num_workers_consume_) {
    HCTR_OWN_THROW(Error_t::OutOfBound, "accomplished_workers_ out of bound!\n");
  }
}
// producer will wait consumers taking away current group
template <typename T>
bool RowGroupReadingThread<T>::wait_until_writeable(bool bypass) {
  int expected_workers = num_workers_consume_;
  auto expected_state = BufferState::ReadyForWrite;
  //!! for debug only
  if (bypass) {
    return true;
  }
  while ((this->accomplished_workers_[worker_id_]->load() != expected_workers) ||
         (this->producer_buffer_stats_[worker_id_]->load() != expected_state)) {
    if (*end_loop_) {
      return false;
    }
    usleep(2);
  }
  this->accomplished_workers_[worker_id_]->store(0);
  this->producer_buffer_stats_[worker_id_]->store(BufferState::Writing);
  return true;
}
template <typename T>
bool RowGroupReadingThread<T>::wait_until_readable(int wait_id) {
  BufferState expected = BufferState::ReadyForRead;
  while (!this->producer_buffer_stats_[wait_id]->compare_exchange_weak(expected,
                                                                       BufferState::ReadyForRead) ||
         this->workers_has_read_[worker_id_ * num_workers_ + wait_id]) {
    expected = BufferState::ReadyForRead;
    if (this->producer_buffer_stats_[wait_id]->load() == BufferState::FileEOF) {
      throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
    }
    if (*end_loop_) {
      return false;
    }
    usleep(2);
  }
  this->workers_has_read_[worker_id_ * num_workers_ + wait_id] = true;
  return true;
}
template <typename T>
void RowGroupReadingThread<T>::reset_read_flag() {
  for (auto i = 0; i < num_workers_; i++) {
    workers_has_read_[i * num_workers_ + worker_id_] = false;
  }
}
template <typename T>
void RowGroupReadingThread<T>::reset_accomplished_worker() {
  this->accomplished_workers_[worker_id_]->store(num_workers_consume_);
}

template <typename T>
bool RowGroupReadingThread<T>::is_eof(int id) {
  if (this->producer_buffer_stats_[id]->load() == BufferState::FileEOF) {
    return true;
  }
  return false;
}

template <typename T>
void RowGroupReadingThread<T>::stop() {
  accomplished_workers_[worker_id_]->store(num_workers_consume_);
  // read escape
  this->producer_buffer_stats_[worker_id_]->store(BufferState::ReadyForRead);
  for (auto i = 0; i < num_workers_; i++) {
    workers_has_read_[worker_id_ * num_workers_ + i] = false;
  }
}
template <typename T>
void RowGroupReadingThread<T>::start() {}

template <typename T>
std::shared_ptr<DFContainer<T>>& RowGroupReadingThread<T>::get_df_container_consumer() {
  return this->df_container_consumer_;
}

template <typename T>
std::shared_ptr<DFContainer<T>>& RowGroupReadingThread<T>::get_df_container_producer(
    int worker_id) {
  return this->df_container_producer_[worker_id];
}
template <typename T>
bool RowGroupReadingThread<T>::source_available() {
  return source_ && source_->is_open();
}

template <typename T>
long long RowGroupReadingThread<T>::get_current_num_row_groups() {
  if (source_ && source_->is_open()) {
    return source_->get_num_row_groups();
  } else {
    HCTR_LOG(ERROR, WORLD, "No row group available, please read_new_file first\n");
    return 0;
  }
}
template <typename T>
long long RowGroupReadingThread<T>::get_local_row_group_id() {
  return this->local_row_group_id_;
}
template <typename T>
void RowGroupReadingThread<T>::set_this_producer_status(BufferState stat) {
  this->producer_buffer_stats_[worker_id_]->store(stat);
};

// params input, others are output
template <typename T>
Error_t RowGroupReadingThread<T>::get_one_read_group(
    const std::vector<DataReaderSparseParam>& params, std::vector<size_t>& dense_dim_array,
    std::vector<int>& one_hot_slot_id, std::vector<int>& sparse_nnz_array) {
  if (!source_->is_open()) {
    return Error_t::EndOfFile;
  }
  auto tbl_w_metadata = source_->read_group(this->local_row_group_id_, this->memory_resource_);
  this->local_row_group_id_ += this->strict_order_of_batches_ ? this->num_workers_ : 1;
  tbl_w_metadata.tbl.swap(this->cached_df_);
  cudf::table_view data_view = cached_df_->view();
  dump_table_data_to(data_view, dense_idx_to_parquet_col_, categorical_idx_parquet_col_, params,
                     df_container_producer_[this->worker_id_], dense_dim_array, one_hot_slot_id,
                     sparse_nnz_array);
  return Error_t::Success;
};

template <typename T>
void RowGroupReadingThread<T>::reset_source(ParquetFileSource* source) {
  this->source_ = source;
  // clear has_read
  reset_read_flag();
  this->accomplished_workers_[worker_id_]->store(num_workers_consume_);
  this->set_this_producer_status(BufferState::ReadyForWrite);
};

template <typename T>
RowGroupReadingThread<T>::~RowGroupReadingThread() {
  this->stop();
}
template <typename T>
void RowGroupReadingThread<T>::read_new_file(long long expected_row_groups) {
  std::set<int> tmp_col_index;
  auto source = this->source_;
  for (int t = 0; t < MAX_TRY; t++) {
    Error_t err = source->next_source(expected_row_groups);

    if (err == Error_t::Success) {
      auto metadata = source->get_file_metadata();
      if (metadata.get_metadata_status()) {
        auto label_col_names = metadata.get_label_names();
        auto dense_col_names = metadata.get_cont_names();
        if (dense_idx_to_parquet_col_.size() != (label_col_names.size() + dense_col_names.size())) {
          int i = 0;
          dense_idx_to_parquet_col_.clear();
          tmp_col_index.clear();
          for (auto& c : label_col_names) {
            tmp_col_index.insert(c.index);
          }
          for (auto it = tmp_col_index.begin(); it != tmp_col_index.end(); it++) {
            dense_idx_to_parquet_col_.insert(std::make_pair(i, *it));
            i++;
          }
          tmp_col_index.clear();
          for (auto& c : dense_col_names) {
            tmp_col_index.insert(c.index);
          }
          for (auto it = tmp_col_index.begin(); it != tmp_col_index.end(); it++) {
            dense_idx_to_parquet_col_.insert(std::make_pair(i, *it));
            i++;
          }
        }
        tmp_col_index.clear();

        auto cat_col_names = metadata.get_cat_names();
        if (categorical_idx_parquet_col_.size() != cat_col_names.size()) {
          categorical_idx_parquet_col_.clear();
          int i = 0;
          for (auto& c : cat_col_names) {
            tmp_col_index.insert(c.index);
          }
          for (auto it = tmp_col_index.begin(); it != tmp_col_index.end(); it++) {
            categorical_idx_parquet_col_.insert(std::make_pair(i, *it));
            i++;
          }
        }
      } else {
        // raise exception
        HCTR_OWN_THROW(Error_t::BrokenFile, "failed to read a file");
      }
      local_row_group_id_ = source->get_row_group();
      return;
    } else if (err == Error_t::EndOfFile) {
      throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
    } else {
      HCTR_OWN_THROW(Error_t::BrokenFile, "failed to read a file");
    }
  }
  HCTR_OWN_THROW(Error_t::BrokenFile, "failed to read a file");
}

template class RowGroupReadingThread<unsigned int>;
template class RowGroupReadingThread<long long>;
}  // namespace HugeCTR

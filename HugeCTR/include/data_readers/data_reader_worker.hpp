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
#include <core23/tensor.hpp>
#include <data_readers/check_none.hpp>
#include <data_readers/check_sum.hpp>
#include <data_readers/csr.hpp>
#include <data_readers/csr23.hpp>
#include <data_readers/data_reader_worker_interface.hpp>
#include <data_readers/file_list.hpp>
#include <data_readers/file_source.hpp>
#include <vector>

namespace HugeCTR {

namespace core23_reader {

template <class T>
class DataReaderWorker : public IDataReaderWorker {
 private:
  DataSetHeader
      data_set_header_;  /**< the header of data set, which has main informations of a data file */
  size_t buffer_length_; /**< buffer size for internal use */
  Check_t check_type_;   /**< check type for data set */
  std::vector<DataReaderSparseParam> params_; /**< configuration of data reader sparse input */
  std::shared_ptr<Checker> checker_; /**< checker aim to perform error check of the input data */
  bool skip_read_{false};            /**< set to true when you want to stop the data reading */
  int current_record_index_{0};
  size_t total_slot_num_;
  std::vector<size_t> last_batch_nnz_;

  core23::Tensor temp_host_dense_buffer_;  // read data to make checker move
  core23::Tensor host_dense_buffer_;
  std::vector<CSR23<T>> host_sparse_buffer_;

  void read_new_file();

  void create_checker() {
    switch (check_type_) {
      case Check_t::Sum:
        checker_ = std::make_shared<CheckSum>(*source_);
        break;
      case Check_t::None:
        checker_ = std::make_shared<CheckNone>(*source_);
        break;
      default:
        assert(!"Error: no such Check_t && should never get here!!");
    }
  }

 public:
  void post_set_source() override {
    create_checker();

    is_eof_ = false;
    buffer23_->state.store(BufferState::ReadyForWrite);
  }

  /**
   * Ctor
   */
  DataReaderWorker(const int worker_id, const int worker_num,
                   const std::shared_ptr<GPUResource>& gpu_resource,
                   const std::shared_ptr<std::atomic<bool>>& loop_flag,
                   const std::shared_ptr<ThreadBuffer23>& buffer, const std::string& file_list,
                   size_t buffer_length, bool repeat, Check_t check_type,
                   const std::vector<DataReaderSparseParam>& params);

  void do_h2d(){};

  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch();

  DataReaderType_t get_reader_type() override { return DataReaderType_t::Norm; }
};

}  // namespace core23_reader

template <class T>
class DataReaderWorker : public IDataReaderWorker {
 private:
  DataSetHeader
      data_set_header_;  /**< the header of data set, which has main informations of a data file */
  size_t buffer_length_; /**< buffer size for internal use */
  Check_t check_type_;   /**< check type for data set */
  std::vector<DataReaderSparseParam> params_; /**< configuration of data reader sparse input */
  std::shared_ptr<Checker> checker_; /**< checker aim to perform error check of the input data */
  bool skip_read_{false};            /**< set to true when you want to stop the data reading */
  int current_record_index_{0};
  size_t total_slot_num_;
  std::vector<size_t> last_batch_nnz_;

  Tensor2<float> temp_host_dense_buffer_;  // read data to make checker move
  Tensor2<float> host_dense_buffer_;
  std::vector<CSR<T>> host_sparse_buffer_;

  void read_new_file();

  void create_checker() {
    switch (check_type_) {
      case Check_t::Sum:
        checker_ = std::make_shared<CheckSum>(*source_);
        break;
      case Check_t::None:
        checker_ = std::make_shared<CheckNone>(*source_);
        break;
      default:
        assert(!"Error: no such Check_t && should never get here!!");
    }
  }

 public:
  void post_set_source() override {
    create_checker();

    is_eof_ = false;
    buffer_->state.store(BufferState::ReadyForWrite);
  }

  /**
   * Ctor
   */
  DataReaderWorker(const int worker_id, const int worker_num,
                   const std::shared_ptr<GPUResource>& gpu_resource,
                   const std::shared_ptr<std::atomic<bool>>& loop_flag,
                   const std::shared_ptr<ThreadBuffer>& buffer, const std::string& file_list,
                   size_t buffer_length, bool repeat, Check_t check_type,
                   const std::vector<DataReaderSparseParam>& params);

  void do_h2d(){};

  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch();

  DataReaderType_t get_reader_type() override { return DataReaderType_t::Norm; }
};

}  // namespace HugeCTR

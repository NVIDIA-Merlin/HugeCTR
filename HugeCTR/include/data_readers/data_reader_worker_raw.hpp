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
#include <data_readers/csr.hpp>
#include <data_readers/csr23.hpp>
#include <data_readers/data_reader_common.hpp>
#include <data_readers/data_reader_worker_interface.hpp>
#include <data_readers/mmap_source.hpp>
#include <fstream>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {

namespace core23_reader {

template <class T>
class DataReaderWorkerRaw : public IDataReaderWorker {
 private:
  std::vector<DataReaderSparseParam> params_; /**< configuration of data reader sparse input */
  bool float_label_dense_;
  size_t total_slot_num_;
  std::vector<size_t> last_batch_nnz_;

  core23::Tensor host_dense_buffer_;
  std::vector<CSR23<T>> host_sparse_buffer_;

  void read_new_file() {
    Error_t flag = source_->next_source(1);
    if (flag == Error_t::EndOfFile) {
      throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
    }
  }

 public:
  void post_set_source() override {
    is_eof_ = false;
    buffer23_->state.store(BufferState::ReadyForWrite);
  }

  /**
   * Ctor
   */
  DataReaderWorkerRaw(const int worker_id, const int worker_num,
                      const std::shared_ptr<GPUResource>& gpu_resource,
                      const std::shared_ptr<std::atomic<bool>>& loop_flag,
                      const std::shared_ptr<ThreadBuffer23>& buffer,
                      std::shared_ptr<MmapOffsetList>& file_offset_list, bool repeat,
                      const std::vector<DataReaderSparseParam>& params, bool float_label_dense);

  void do_h2d(){};
  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch();

  DataReaderType_t get_reader_type() override { return DataReaderType_t::Raw; }
};
}  // namespace core23_reader

template <class T>
class DataReaderWorkerRaw : public IDataReaderWorker {
 private:
  std::vector<DataReaderSparseParam> params_; /**< configuration of data reader sparse input */
  bool float_label_dense_;
  size_t total_slot_num_;
  std::vector<size_t> last_batch_nnz_;

  Tensor2<float> host_dense_buffer_;
  std::vector<CSR<T>> host_sparse_buffer_;

  void read_new_file() {
    Error_t flag = source_->next_source(1);
    if (flag == Error_t::EndOfFile) {
      throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
    }
  }

 public:
  void post_set_source() override {
    is_eof_ = false;
    buffer_->state.store(BufferState::ReadyForWrite);
  }

  /**
   * Ctor
   */
  DataReaderWorkerRaw(const int worker_id, const int worker_num,
                      const std::shared_ptr<GPUResource>& gpu_resource,
                      const std::shared_ptr<std::atomic<bool>>& loop_flag,
                      const std::shared_ptr<ThreadBuffer>& buffer,
                      std::shared_ptr<MmapOffsetList>& file_offset_list, bool repeat,
                      const std::vector<DataReaderSparseParam>& params, bool float_label_dense);

  void do_h2d(){};
  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch();

  DataReaderType_t get_reader_type() override { return DataReaderType_t::Raw; }
};
}  // namespace HugeCTR

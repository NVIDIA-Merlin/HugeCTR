/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <fstream>
#include <vector>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/csr.hpp"
#include "HugeCTR/include/csr_chunk.hpp"
#include "HugeCTR/include/file_list.hpp"
#include "HugeCTR/include/heap.hpp"

namespace HugeCTR {

/**
 * @brief worker of data reader.
 *
 * This is the data readers which will be initilized within multiple
 * threads in class DataReader.
 * It reads data from files in file_list to CSR heap.
 */
template <class T>
class DataReaderMultiThreads {
 private:
  FileList& file_list_;         /**< file list of data set */
  Heap<CSRChunk<T>>& csr_heap_; /**< heap to cache the data set */
  DataSetHeader
      data_set_header_; /**< the header of data set, which has main informations of a data file */
  long long current_record_index_{0}; /**< the index of current reading record in a data file */
  std::ifstream in_file_stream_;      /**< file stream of data set file */
  std::string file_name_;             /**< file name of current file */
  std::unique_ptr<T[]> feature_ids_;  /**< a buffer to cache the readed feature from data set */
  size_t buffer_length_;              /**< max possible nnz in a slot */
  bool skip_read_{false};             /**< set to true when you want to stop the data reading */

  /**
   * Open data file and read header+
   */
  void open_file_and_read_head() {
    std::string file_name = file_list_.get_a_file();
    in_file_stream_.open(file_name, std::ifstream::binary);
    if (!in_file_stream_.is_open()) {
      CK_THROW_(Error_t::FileCannotOpen, "in_file_stream_.is_open() failed: " + file_name);
    }

    in_file_stream_.read(reinterpret_cast<char*>(&data_set_header_), sizeof(DataSetHeader));
#ifndef NDEBUG
    std::cout << file_name << std::endl;
    std::cout << "number_of_records:" << data_set_header_.number_of_records
              << ", label_dim:" << data_set_header_.label_dim
              << ", slot_num:" << data_set_header_.slot_num << std::endl;
#endif

    current_record_index_ = 0;
    if (!(data_set_header_.number_of_records > 0)) {
      CK_THROW_(Error_t::WrongInput, "number_of_records <= 0");
    }
  }

 public:
  /**
   * Ctor
   */
  DataReaderMultiThreads(Heap<CSRChunk<T>>& csr_heap, FileList& file_list, size_t buffer_length)
      : file_list_(file_list),
        csr_heap_(csr_heap),
        feature_ids_(new T[buffer_length]()),
        buffer_length_(buffer_length){};

  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch();

  /**
   * skip data reading in read_a_batch()
   */
  void skip_read() { skip_read_ = true; }
};

template <class T>
void DataReaderMultiThreads<T>::read_a_batch() {
  try {
    if (!in_file_stream_.is_open()) {
      open_file_and_read_head();
    }
    unsigned int key = 0;
    CSRChunk<T>* chunk_tmp = nullptr;
    csr_heap_.free_chunk_checkout(&chunk_tmp, &key);
    if (!skip_read_) {
      std::vector<CSR<T>>& csr_buffers = chunk_tmp->get_csr_buffers();
      const std::vector<PinnedBuffer<float>>& label_buffers = chunk_tmp->get_label_buffers();
      const int label_dim = chunk_tmp->get_label_dim();
      if (data_set_header_.label_dim != label_dim) {
        CK_THROW_(Error_t::WrongInput, "data_set_header_.label_dim != label_dim");
      }

      std::unique_ptr<int[]> label(new int[label_dim]);
      for (auto& csr_buffer : csr_buffers) {
        csr_buffer.reset();
      }
      assert(label_buffers.size() > 0);
      for (int i = 0; i < chunk_tmp->get_batchsize(); i++) {
        in_file_stream_.read(reinterpret_cast<char*>(label.get()), sizeof(int) * (label_dim));
        {
          // We suppose that the data parallel mode is like this
          int buffer_id = i / (chunk_tmp->get_batchsize() / label_buffers.size());
          assert(buffer_id < static_cast<int>(label_buffers.size()));
          int local_id = i % (chunk_tmp->get_batchsize() / static_cast<int>(label_buffers.size()));
          assert(local_id < (chunk_tmp->get_batchsize() / static_cast<int>(label_buffers.size())));
          for (int j = 0; j < label_dim; j++) {
            // row major for label buffer
            label_buffers[buffer_id][local_id * label_dim + j] = label[j];
          }
        }

        for (int k = 0; k < data_set_header_.slot_num; k++) {
          for (auto& csr_buffer : csr_buffers) {
            csr_buffer.new_row();
          }
          int nnz;
          in_file_stream_.read(reinterpret_cast<char*>(&nnz), sizeof(int));
          if (nnz > (int)buffer_length_ || nnz < 0) {
            ERROR_MESSAGE_("nnz > buffer_length_ | nnz < 0");
          }

#ifndef NDEBUG
          if (i == 0)
            std::cout << "[HCDEBUG]"
                      << "nnz: " << nnz << std::endl;
#endif

          in_file_stream_.read(reinterpret_cast<char*>(feature_ids_.get()), sizeof(T) * nnz);
          for (int j = 0; j < nnz; j++) {
            // We suppose that the module parallel mode is like this
            int buffer_id = feature_ids_[j] % csr_buffers.size();
            T local_id = feature_ids_[j];
            assert(buffer_id < static_cast<int>(csr_buffers.size()));
            csr_buffers[buffer_id].push_back(local_id);
#ifndef NDEBUG
            if (i == 0)
              std::cout << "[HCDEBUG]"
                        << "feature_ids:" << feature_ids_[j] << " local_id: " << local_id
                        << std::endl;
#endif
          }
        }
        current_record_index_++;
        // start a new file when finish one file read
        if (current_record_index_ >= data_set_header_.number_of_records) {
          in_file_stream_.close();
          open_file_and_read_head();
        }
      }
      for (auto& csr_buffer : csr_buffers) {
        csr_buffer.new_row();
      }
    }
    csr_heap_.chunk_write_and_checkin(key);
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  return;
}

}  // namespace HugeCTR

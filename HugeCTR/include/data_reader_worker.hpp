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
#include <check_none.hpp>
#include <check_sum.hpp>
#include <common.hpp>
#include <csr.hpp>
#include <csr_chunk.hpp>
#include <data_reader_worker_interface.hpp>
#include <file_list.hpp>
#include <file_source.hpp>
#include <fstream>
#include <heapex.hpp>
#include <vector>

namespace HugeCTR {
template <class T>
class DataReaderWorker : public IDataReaderWorker {
 private:
  const unsigned int worker_id_{0};
  const unsigned int worker_num_{0};
  std::shared_ptr<HeapEx<CSRChunk<T>>> csr_heap_; /**< heap to cache the data set */
  DataSetHeader
      data_set_header_;  /**< the header of data set, which has main informations of a data file */
  size_t buffer_length_; /**< buffer size for internal use */
  Check_t check_type_;   /**< check type for data set */
  std::vector<DataReaderSparseParam> params_; /**< configuration of data reader sparse input */
  T* feature_ids_;                   /**< a buffer to cache the readed feature from data set */
  std::shared_ptr<Source> source_;   /**< source: can be file or network */
  std::shared_ptr<Checker> checker_; /**< checker aim to perform error check of the input data */
  bool skip_read_{false};            /**< set to true when you want to stop the data reading */
  const int MAX_TRY = 10;
  int current_record_index_{0};
  int slots_{0};
  void read_new_file() {
    for (int i = 0; i < MAX_TRY; i++) {
      checker_->next_source();

      Error_t err =
          checker_->read(reinterpret_cast<char*>(&data_set_header_), sizeof(DataSetHeader));
      current_record_index_ = 0;
      if (!(data_set_header_.error_check == 0 && check_type_ == Check_t::None) &&
          !(data_set_header_.error_check == 1 && check_type_ == Check_t::Sum)) {
        ERROR_MESSAGE_("DataHeaderError");
        continue;
      }
      if (data_set_header_.slot_num != slots_) {
        ERROR_MESSAGE_("DataHeaderError");
        continue;
      }
      if (err == Error_t::Success) {
        return;
      }
    }
    CK_THROW_(Error_t::BrokenFile, "failed to read a file");
  }

 public:
  /**
   * Ctor
   */
  DataReaderWorker(unsigned int worker_id, unsigned int worker_num,
                   const std::shared_ptr<HeapEx<CSRChunk<T>>>& csr_heap,
                   const std::string& file_list, size_t buffer_length, Check_t check_type,
                   const std::vector<DataReaderSparseParam>& params)
      : worker_id_(worker_id),
        worker_num_(worker_num),
        csr_heap_(csr_heap),
        buffer_length_(buffer_length),
        check_type_(check_type),
        params_(params),
        feature_ids_(new T[buffer_length]()) {
    if (worker_id >= worker_num) {
      CK_THROW_(Error_t::BrokenFile, "DataReaderWorker: worker_id >= worker_num");
    }
    slots_ = 0;
    for (auto& p : params) {
      slots_ += p.slot_num;
    }
    source_ = std::make_shared<FileSource>(worker_id, worker_num, file_list);
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
  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch();

  /**
   * skip data reading in read_a_batch()
   */
  void skip_read() { skip_read_ = true; }
};

#define CK_READ_(x)                                        \
  do {                                                     \
    Error_t __ERR = (x);                                   \
    if (__ERR == Error_t::Success) {                       \
    } else if (__ERR == Error_t::DataCheckError) {         \
      csr_chunk->apply_to_csr_buffers(&CSR<T>::roll_back); \
      i--;                                                 \
      ERROR_MESSAGE_("Error_t::DataCheckError");           \
      goto END_SAMPLE;                                     \
    } else {                                               \
      csr_chunk->apply_to_csr_buffers(&CSR<T>::roll_back); \
      i--;                                                 \
      read_new_file();                                     \
      goto END_SAMPLE;                                     \
    }                                                      \
  } while (0)

template <class T>
void DataReaderWorker<T>::read_a_batch() {
  try {
    if (!checker_->is_open()) {
      read_new_file();
    }
    CSRChunk<T>* csr_chunk = nullptr;
    csr_heap_->free_chunk_checkout(&csr_chunk, worker_id_);

    if (!skip_read_) {
      csr_chunk->set_current_batchsize(csr_chunk->get_batchsize());
      Tensors2<float>& label_dense_buffers = csr_chunk->get_label_buffers();
      const int label_dense_dim = csr_chunk->get_label_dense_dim();
      if (data_set_header_.label_dim + data_set_header_.dense_dim != label_dense_dim)
        CK_THROW_(Error_t::WrongInput,
                  "data_set_header_.label_dim + data_set_header_.dense_dim != label_dense_dim");
      std::unique_ptr<float[]> label_dense(new float[label_dense_dim]());

      csr_chunk->apply_to_csr_buffers(&CSR<T>::reset);
      assert(label_dense_buffers.size() > 0);
      // batch loop
      for (int i = 0; i < csr_chunk->get_batchsize(); i++) {
        int param_id = 0;
        csr_chunk->apply_to_csr_buffers(&CSR<T>::set_check_point);

        CK_READ_(checker_->read(reinterpret_cast<char*>(label_dense.get()),
                                sizeof(float) * label_dense_dim));

        {
          // We suppose that the data parallel mode is like this
          // The subsequence samples will be located to the same GPU
          int buffer_id = i / (csr_chunk->get_batchsize() / label_dense_buffers.size());
          assert((unsigned int)buffer_id < label_dense_buffers.size());
          int local_id = i % (csr_chunk->get_batchsize() / label_dense_buffers.size());
          assert((unsigned int)local_id <
                 (csr_chunk->get_batchsize() / label_dense_buffers.size()));
          float* ptr = label_dense_buffers[buffer_id].get_ptr();
          for (int j = 0; j < label_dense_dim; j++) {
            ptr[local_id * label_dense_dim + j] = label_dense[j];  // row major for label buffer
          }
        }

        for (auto& param : params_) {
          for (int k = 0; k < param.slot_num; k++) {
            int nnz;
            CK_READ_(checker_->read(reinterpret_cast<char*>(&nnz), sizeof(int)));

            if (nnz > (int)buffer_length_ || nnz < 0) {
              ERROR_MESSAGE_("nnz > buffer_length_ | nnz < 0");
            }

            CK_READ_(checker_->read(reinterpret_cast<char*>(feature_ids_), sizeof(T) * nnz));
            if (param.type == DataReaderSparse_t::Distributed) {
              for (int dev_id = 0; dev_id < csr_chunk->get_num_devices(); dev_id++) {
                csr_chunk->get_csr_buffer(param_id, dev_id).new_row();
              }
              for (int j = 0; j < nnz; j++) {
                int dev_id = feature_ids_[j] % csr_chunk->get_num_devices();
                dev_id = std::abs(dev_id);
                T local_id = feature_ids_[j];
                assert(dev_id < csr_chunk->get_num_devices());
                /* #ifndef NDEBUG
                                if (i >= 0)
                                  std::cout << "[HCDEBUG]"
                                            << "feature_ids:" << feature_ids_[j] << " local_id: " <<
                local_id
                                            << " param_id: " << param_id << " dev_id: " << dev_id <<
                std::endl;
                #endif */

                csr_chunk->get_csr_buffer(param_id, dev_id).push_back(local_id);
              }
            } else if (param.type == DataReaderSparse_t::Localized) {
              int dev_id = k % csr_chunk->get_num_devices();
              csr_chunk->get_csr_buffer(param_id, dev_id).new_row();
              for (int j = 0; j < nnz; j++) {
                T local_id = feature_ids_[j];
                csr_chunk->get_csr_buffer(param_id, dev_id).push_back(local_id);
              }
            } else {
              CK_THROW_(Error_t::UnspecificError, "param.type is not defined");
            }
          }
          param_id++;
        }  // for(auto& param: params_)
      END_SAMPLE:;

        current_record_index_++;

        // start a new file when finish one file read
        if (current_record_index_ >= data_set_header_.number_of_records) {
          read_new_file();
        }
      }  // batch loop
      // write the last index to row
      csr_chunk->apply_to_csr_buffers(&CSR<T>::new_row);
    }
    csr_heap_->chunk_write_and_checkin(worker_id_);
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  return;
}

}  // namespace HugeCTR

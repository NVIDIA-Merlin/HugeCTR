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
#include <common.hpp>
#include <data_readers/check_none.hpp>
#include <data_readers/check_sum.hpp>
#include <data_readers/chunk_producer.hpp>
#include <data_readers/csr.hpp>
#include <data_readers/csr_chunk.hpp>
#include <data_readers/data_reader_worker_interface.hpp>
#include <data_readers/file_list.hpp>
#include <data_readers/file_source.hpp>
#include <data_readers/heapex.hpp>
#include <fstream>
#include <vector>

namespace HugeCTR {
template <class T>
class DataReaderWorker : public IDataReaderWorker {
 private:
  const unsigned int worker_id_{0};
  const unsigned int worker_num_{0};
  std::shared_ptr<ChunkProducer<CSRChunk<T>>> csr_heap_; /**< heap to cache the data set */
  DataSetHeader
      data_set_header_;  /**< the header of data set, which has main informations of a data file */
  size_t buffer_length_; /**< buffer size for internal use */
  Check_t check_type_;   /**< check type for data set */
  std::vector<DataReaderSparseParam> params_; /**< configuration of data reader sparse input */
  T* feature_ids_;                   /**< a buffer to cache the readed feature from data set */
  std::shared_ptr<Checker> checker_; /**< checker aim to perform error check of the input data */
  bool skip_read_{false};            /**< set to true when you want to stop the data reading */
  const int MAX_TRY = 10;
  int current_record_index_{0};
  int slots_{0};

  // TODO(minseokl, 11062020): they must be moved to the parent class if the EOF is enabled
  // in the other workers such as Parquet and Raw.
  std::condition_variable eof_cv_;
  std::mutex eof_mtx_;

  void read_new_file() {
    for (int i = 0; i < MAX_TRY; i++) {
      if (checker_->next_source() == Error_t::EndOfFile) {
        throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
      }

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

  void post_set_source() override {
    create_checker();
    eof_cv_.notify_all();
    is_eof_ = false;
  }

 public:
  /**
   * Ctor
   */
  DataReaderWorker(unsigned int worker_id, unsigned int worker_num,
                   const std::shared_ptr<ChunkProducer<CSRChunk<T>>>& csr_heap,
                   const std::string& file_list, size_t buffer_length, bool repeat,
                   Check_t check_type, const std::vector<DataReaderSparseParam>& params)
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
    source_ = std::make_shared<FileSource>(worker_id, worker_num, file_list, repeat);
    // In the no-repeat mode, the data reader worker doesn't start from the beginning.
    // Thus, whe constructed, it is considered as the same as the EOF state,
    // so that set_*_source can be done on the client code side.
    if (!repeat) {
      is_eof_ = true;
    }
    create_checker();
  }

  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch();

  /**
   * skip data reading in read_a_batch()
   */
  void skip_read() {
    skip_read_ = true;
    eof_cv_.notify_all();
  }
};

template <class T>
void DataReaderWorker<T>::read_a_batch() {
  int i = 0;
  CSRChunk<T>* csr_chunk = nullptr;
  try {
    if (!checker_->is_open()) {
      read_new_file();
    }
    csr_chunk = csr_heap_->checkout_free_chunk(worker_id_);

    if (!skip_read_) {
      // if the EOF is faced, the current batch size can be changed later
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
      for (i = 0; i < csr_chunk->get_batchsize(); i++) {
        try {
          int param_id = 0;
          csr_chunk->apply_to_csr_buffers(&CSR<T>::set_check_point);

          CK_THROW_(checker_->read(reinterpret_cast<char*>(label_dense.get()),
                                   sizeof(float) * label_dense_dim),
                    "failure in reading label_dense");

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
              CK_THROW_(checker_->read(reinterpret_cast<char*>(&nnz), sizeof(int)),
                        "failure in reading nnz");

              if (nnz > (int)buffer_length_ || nnz < 0) {
                ERROR_MESSAGE_("nnz > buffer_length_ | nnz < 0");
              }

              CK_THROW_(checker_->read(reinterpret_cast<char*>(feature_ids_), sizeof(T) * nnz),
                        "failure in reading feature_ids_");
              if (param.type == DataReaderSparse_t::Distributed) {
                for (int dev_id = 0; dev_id < csr_chunk->get_num_devices(); dev_id++) {
                  csr_chunk->get_csr_buffer(param_id, dev_id).new_row();
                }
                for (int j = 0; j < nnz; j++) {
                  int dev_id = feature_ids_[j] % csr_chunk->get_num_devices();
                  dev_id = std::abs(dev_id);
                  T local_id = feature_ids_[j];
                  assert(dev_id < csr_chunk->get_num_devices());
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
        } catch (const internal_runtime_error& rt_err) {
          i--;  // restart i-th sample
          csr_chunk->apply_to_csr_buffers(&CSR<T>::roll_back);
          Error_t err = rt_err.get_error();
          if (err == Error_t::DataCheckError) {
            ERROR_MESSAGE_("Error_t::DataCheckError");
          } else {            // Error_t::BrokenFile, Error_t::UnspecificEror, ...
            read_new_file();  // can throw Error_t::EOF
          }
        } catch (const std::runtime_error& rt_err) {
          std::cerr << rt_err.what() << std::endl;
          throw;
        }

        current_record_index_++;

        // start a new file when finish one file read
        if (current_record_index_ >= data_set_header_.number_of_records) {
          read_new_file();  // can throw Error_t::EOF
        }
      }  // batch loop
      // write the last index to row
      csr_chunk->apply_to_csr_buffers(&CSR<T>::new_row);
    }
    csr_heap_->commit_data_chunk(worker_id_, false);
  } catch (const internal_runtime_error& rt_err) {
    Error_t err = rt_err.get_error();
    if (err == Error_t::EndOfFile) {
      if (csr_chunk != nullptr && i > 0) {
        // it faced the EOF after the last sample was processed successfully,
        if (current_record_index_ >= data_set_header_.number_of_records) {
          i++;
        }
        csr_chunk->set_current_batchsize(i);
        for (int j = i; j < csr_chunk->get_batchsize(); j++) {
          fill_empty_sample(params_, csr_chunk);
        }
        // write the last index to row
        csr_chunk->apply_to_csr_buffers(&CSR<T>::new_row);
        // push the partially filled batch
        csr_heap_->commit_data_chunk(worker_id_, false);
      } else {
        // push nop to singal to DataCollector that it is the EOF
        csr_heap_->commit_data_chunk(worker_id_, true);
        is_eof_ = true;
        std::unique_lock<std::mutex> lock(eof_mtx_);
        // wait for the new source is set
        eof_cv_.wait(lock);
      }
    } else {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

}  // namespace HugeCTR

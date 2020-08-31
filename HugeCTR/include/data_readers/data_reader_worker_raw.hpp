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
#include <common.hpp>
#include <csr.hpp>
#include <csr_chunk.hpp>
#include <data_readers/data_reader_worker_interface.hpp>
#include <fstream>
#include <heapex.hpp>
#include <mmap_source.hpp>
#include <vector>

namespace HugeCTR {
template <class T>
class DataReaderWorkerRaw : public IDataReaderWorker {
 private:
  const unsigned int worker_id_{0};
  const unsigned int worker_num_{0};
  std::shared_ptr<HeapEx<CSRChunk<T>>> csr_heap_; /**< heap to cache the data set */
  std::vector<DataReaderSparseParam> params_;     /**< configuration of data reader sparse input */
  int* feature_ids_;               /**< a buffer to cache the readed feature from data set */
  std::shared_ptr<Source> source_; /**< source: can be file or network */
  bool skip_read_{false};          /**< set to true when you want to stop the data reading */
  const int MAX_TRY = 10;
  int slots_{0};
  const std::vector<long long> slot_offset_;
  const int label_dim_{1};
  const bool float_label_dense_;

  void read_new_file() { source_->next_source(); }
  //  std::vector<int> data_buffer_; /**< data buffer with size of full batchsize*/
 public:
  /**
   * Ctor
   */
  DataReaderWorkerRaw(unsigned int worker_id, unsigned int worker_num,
                      std::shared_ptr<MmapOffsetList>& file_offset_list,
                      const std::shared_ptr<HeapEx<CSRChunk<T>>>& csr_heap,
                      const std::string& file_name,
                      const std::vector<DataReaderSparseParam>& params,
                      const std::vector<long long>& slot_offset, int label_dim,
                      bool float_label_dense)
      : worker_id_(worker_id),
        worker_num_(worker_num),
        csr_heap_(csr_heap),
        params_(params),
        slot_offset_(slot_offset),
        label_dim_(label_dim),
        float_label_dense_(float_label_dense) {
    if (worker_id >= worker_num) {
      CK_THROW_(Error_t::BrokenFile, "DataReaderWorkerRaw: worker_id >= worker_num");
    }
    slots_ = 0;
    for (auto& p : params) {
      slots_ += p.slot_num;
    }
    if (slots_ != (int)slot_offset_.size() && !slot_offset_.empty()) {
      CK_THROW_(Error_t::WrongInput, "DataReaderWorkerRaw: slots_ != slot_offset_.size()");
    }
    feature_ids_ = new int[slots_]();

    source_ = std::make_shared<MmapSource>(file_offset_list, worker_id);
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

template <class T>
void DataReaderWorkerRaw<T>::read_a_batch() {
  try {
    read_new_file();

    CSRChunk<T>* csr_chunk = nullptr;
    csr_heap_->free_chunk_checkout(&csr_chunk, worker_id_);

    if (!skip_read_ && csr_chunk != nullptr) {
      long long current_batchsize = source_->get_num_of_items_in_source();
      if (current_batchsize != csr_chunk->get_batchsize()) {
        std::cout << "current_batchsize: " << current_batchsize
                  << "batchsize: " << csr_chunk->get_batchsize() << std::endl;
      }
      csr_chunk->set_current_batchsize(current_batchsize);
      Tensors2<float>& label_dense_buffers = csr_chunk->get_label_buffers();

      int slot_id = 0;

      csr_chunk->apply_to_csr_buffers(&CSR<T>::reset);
      assert(label_dense_buffers.size() > 0);

      size_t num_slots = 0;
      for (auto& param : params_) {
        num_slots += param.slot_num;
      }
      const int label_dense_dim = csr_chunk->get_label_dense_dim();
      size_t label_dense_length =
          label_dense_dim * (float_label_dense_ ? sizeof(float) : sizeof(int));
      size_t sample_length = num_slots * sizeof(int) + label_dense_length;

      char* data_buffer = source_->get_ptr();

      // batch loop
      for (int i = 0; i < csr_chunk->get_batchsize(); i++) {
        // to compensate the csr when current_batchsize != csr_chunk->get_batchsize()
        char* sample_cur = data_buffer + sample_length * i;
        if (i >= current_batchsize) {
          int param_id = 0;
          for (auto& param : params_) {
            for (int k = 0; k < param.slot_num; k++) {
              if (param.type == DataReaderSparse_t::Distributed) {
                for (int dev_id = 0; dev_id < csr_chunk->get_num_devices(); dev_id++) {
                  csr_chunk->get_csr_buffer(param_id, dev_id).new_row();
                }
              } else if (param.type == DataReaderSparse_t::Localized) {
                int dev_id = k % csr_chunk->get_num_devices();
                csr_chunk->get_csr_buffer(param_id, dev_id).new_row();
              } else {
                CK_THROW_(Error_t::UnspecificError, "param.type is not defined");
              }
            }
            param_id++;
          }  // for(auto& param: params_)
          continue;
        }  // if(i>= current_batchsize)

        int param_id = 0;
        csr_chunk->apply_to_csr_buffers(&CSR<T>::set_check_point);

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
            if (j < label_dim_) {
              // label buffer is in row-major layout
              ptr[local_id * label_dense_dim + j] = float_label_dense_
                                                        ? reinterpret_cast<float*>(sample_cur)[j]
                                                        : reinterpret_cast<int*>(sample_cur)[j];
            } else {
              // if the underlying value is int, do DLRM-style preprocessing
              // otherwise, the value is just directly used.
              float val = float_label_dense_ ? reinterpret_cast<float*>(sample_cur)[j]
                                             : log(reinterpret_cast<int*>(sample_cur)[j] + 1.f);
              ptr[local_id * label_dense_dim + j] = val;
            }
          }
          // if(local_id == 0)
          //   std::cout << std::endl;
        }

        int* feature_ids = reinterpret_cast<int*>(sample_cur + label_dense_length);
        if (params_.size() == 1 && params_[0].type == DataReaderSparse_t::Localized &&
            !slot_offset_.empty()) {
          auto& param = params_[0];
          for (int k = 0; k < param.slot_num; k++) {
            int dev_id = k % csr_chunk->get_num_devices();
            T local_id = feature_ids[k] + slot_offset_[k];
            csr_chunk->get_csr_buffer(param_id, dev_id).push_back_new_row(local_id);
          }
        } else {
          slot_id = 0;
          for (auto& param : params_) {
            for (int k = 0; k < param.slot_num; k++) {
              long long slot_offset = slot_offset_.empty() ? 0 : slot_offset_[slot_id];
              if (param.type == DataReaderSparse_t::Distributed) {
                for (int dev_id = 0; dev_id < csr_chunk->get_num_devices(); dev_id++) {
                  csr_chunk->get_csr_buffer(param_id, dev_id).new_row();
                }
                {
                  T local_id = feature_ids[k] + slot_offset;
                  int dev_id = local_id % csr_chunk->get_num_devices();
                  dev_id = std::abs(dev_id);

                  assert(dev_id < csr_chunk->get_num_devices());
                  /* #ifndef NDEBUG
                                    if (i >= 0)
                                      std::cout << "[HCDEBUG]"
                                                << "feature_ids:" << feature_ids[k] << " local_id: "
                  << local_id
                                                << " param_id: " << param_id << " dev_id: " <<
                  dev_id << std::endl;
                  #endif */

                  csr_chunk->get_csr_buffer(param_id, dev_id).push_back(local_id);
                }
              } else if (param.type == DataReaderSparse_t::Localized) {
                int dev_id = k % csr_chunk->get_num_devices();
                csr_chunk->get_csr_buffer(param_id, dev_id).new_row();
                T local_id = feature_ids[k] + slot_offset;
                csr_chunk->get_csr_buffer(param_id, dev_id).push_back(local_id);

              } else {
                CK_THROW_(Error_t::UnspecificError, "param.type is not defined");
              }
              slot_id++;
            }
            feature_ids += param.slot_num;
            param_id++;
          }  // for(auto& param: params_)
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

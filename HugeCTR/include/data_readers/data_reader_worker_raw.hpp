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
#include <data_readers/csr.hpp>
#include <data_readers/data_reader_worker_interface.hpp>
#include <data_readers/mmap_source.hpp>
#include <fstream>
#include <vector>

#include "data_readers/data_reader_common.hpp"
#include "tensor2.hpp"

namespace HugeCTR {
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
    Error_t flag = source_->next_source();
    if (flag == Error_t::EndOfFile) {
      throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
    }
  }

  void post_set_source() override {
    is_eof_ = false;
    buffer_->state.store(BufferState::ReadyForWrite);
  }

 public:
  /**
   * Ctor
   */
  DataReaderWorkerRaw(const int worker_id, const int worker_num,
                      const std::shared_ptr<GPUResource>& gpu_resource, int* loop_flag,
                      const std::shared_ptr<ThreadBuffer>& buffer,
                      std::shared_ptr<MmapOffsetList>& file_offset_list, bool repeat,
                      const std::vector<DataReaderSparseParam>& params, bool float_label_dense)
      : IDataReaderWorker(worker_id, worker_num, gpu_resource, !repeat, loop_flag, buffer),
        params_(params),
        float_label_dense_(float_label_dense),
        total_slot_num_(0),
        last_batch_nnz_(params.size(), 0) {
    CudaCPUDeviceContext ctx(gpu_resource->get_device_id());
    
    if (worker_id >= worker_num) {
      CK_THROW_(Error_t::BrokenFile, "DataReaderWorkerRaw: worker_id >= worker_num");
    }

    source_ = std::make_shared<MmapSource>(file_offset_list, worker_id);

    int batch_size = buffer->batch_size;
    int batch_size_start_idx = buffer->batch_size_start_idx;
    int batch_size_end_idx = buffer->batch_size_end_idx;
    int label_dim = buffer->label_dim;
    int dense_dim = buffer->dense_dim;

    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> buff =
        GeneralBuffer2<CudaHostAllocator>::create();

    buff->reserve({static_cast<size_t>(batch_size_end_idx - batch_size_start_idx),
                   static_cast<size_t>(label_dim + dense_dim)},
                  &host_dense_buffer_);

    for (auto& param : params) {
      host_sparse_buffer_.emplace_back(batch_size * param.slot_num,
                                       batch_size * param.max_feature_num);
    }

    buff->allocate();
    for (auto& param : params) {
      total_slot_num_ += param.slot_num;
    }
  }
  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch() {
    try {
      read_new_file();
    } catch (const internal_runtime_error& rt_err) {
      Error_t err = rt_err.get_error();
      if (err == Error_t::EndOfFile) {
        if (!wait_until_h2d_ready()) return;
        buffer_->current_batch_size = 0;
        assert(buffer_->state.load() == BufferState::Writing);
        is_eof_ = true;
        buffer_->state.store(BufferState::ReadyForRead);

        while (buffer_->state.load() != BufferState::ReadyForWrite) {
          usleep(2);
          if (*loop_flag_ == 0) return;
        }
        return;
      } else {
        throw;
      }
    }

    long long current_batchsize = source_->get_num_of_items_in_source();
    if (current_batchsize != buffer_->batch_size) {
      std::cout << "current_batchsize: " << current_batchsize
                << "batchsize: " << buffer_->batch_size << std::endl;
    }

    char* data_buffer = source_->get_ptr();
    int label_dim = buffer_->label_dim;
    int dense_dim = buffer_->dense_dim;
    int label_dense_dim = label_dim + dense_dim;
    int batch_size_start_idx = buffer_->batch_size_start_idx;
    int batch_size_end_idx = buffer_->batch_size_end_idx;
    size_t label_dense_length =
        label_dense_dim * (float_label_dense_ ? sizeof(float) : sizeof(int));
    size_t sample_length = total_slot_num_ * sizeof(int) + label_dense_length;

    for (auto& each_csr : host_sparse_buffer_) {
      each_csr.reset();
    }
    for (int batch_idx = 0; batch_idx < buffer_->batch_size; ++batch_idx) {
      if (batch_idx >= current_batchsize) {
        for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
          auto& param = params_[param_id];
          auto& current_csr = host_sparse_buffer_[param_id];
          for (int k = 0; k < param.slot_num; k++) {
            current_csr.new_row();
          }
        }
        if (batch_idx >= batch_size_start_idx &&
            batch_idx < batch_size_end_idx) {  // only read local device dense data
          float* ptr =
              host_dense_buffer_.get_ptr() + (batch_idx - batch_size_start_idx) * label_dense_dim;

          for (int j = 0; j < label_dense_dim; j++) {
            ptr[j] = 0.f;
          }
        }
        continue;
      }
      char* sample_cur = data_buffer + sample_length * batch_idx;

      if (batch_idx >= batch_size_start_idx &&
          batch_idx < batch_size_end_idx) {  // only read local device dense data
        float* ptr =
            host_dense_buffer_.get_ptr() + (batch_idx - batch_size_start_idx) * label_dense_dim;

        for (int j = 0; j < label_dense_dim; j++) {
          if (j < label_dim) {
            // label buffer is in row-major layout
            ptr[j] = float_label_dense_ ? reinterpret_cast<float*>(sample_cur)[j]
                                        : reinterpret_cast<int*>(sample_cur)[j];
          } else {
            // if the underlying value is int, do DLRM-style preprocessing
            // otherwise, the value is just directly used.
            float val = float_label_dense_ ? reinterpret_cast<float*>(sample_cur)[j]
                                           : log(reinterpret_cast<int*>(sample_cur)[j] + 1.f);
            ptr[j] = val;
          }
        }
      }
      {
        int* feature_ids = reinterpret_cast<int*>(sample_cur + label_dense_length);

        for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
          auto& param = params_[param_id];
          auto& current_csr = host_sparse_buffer_[param_id];
          for (int k = 0; k < param.slot_num; k++) {
            current_csr.push_back_new_row(feature_ids[k]);
          }
          feature_ids += param.slot_num;
        }
      }
    }
    for (auto& each_csr : host_sparse_buffer_) {
      each_csr.new_row();
    }

    // do h2d
    // wait buffer and schedule
    if (!wait_until_h2d_ready()) return;
    buffer_->current_batch_size = current_batchsize;
    {
      CudaCPUDeviceContext context(gpu_resource_->get_device_id());
      auto dst_dense_tensor = Tensor2<float>::stretch_from(buffer_->device_dense_buffers);
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_dense_tensor.get_ptr(), host_dense_buffer_.get_ptr(),
                                     host_dense_buffer_.get_size_in_bytes(), cudaMemcpyHostToDevice,
                                     gpu_resource_->get_memcpy_stream()));

      for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
        auto dst_sparse_tensor =
            SparseTensor<T>::stretch_from(buffer_->device_sparse_buffers[param_id]);
        if (buffer_->is_fixed_length[param_id] &&
            last_batch_nnz_[param_id] == host_sparse_buffer_[param_id].get_num_values()) {
          CK_CUDA_THROW_(cudaMemcpyAsync(dst_sparse_tensor.get_value_ptr(),
                                         host_sparse_buffer_[param_id].get_value_tensor().get_ptr(),
                                         host_sparse_buffer_[param_id].get_num_values() * sizeof(T),
                                         cudaMemcpyHostToDevice,
                                         gpu_resource_->get_memcpy_stream()));
        } else {
          sparse_tensor_helper::cuda::copy_async(dst_sparse_tensor, host_sparse_buffer_[param_id],
                                                 gpu_resource_->get_memcpy_stream());
          last_batch_nnz_[param_id] = host_sparse_buffer_[param_id].get_num_values();
        }
      }
      CK_CUDA_THROW_(cudaStreamSynchronize(gpu_resource_->get_memcpy_stream()));
    }

    assert(buffer_->state.load() == BufferState::Writing);
    buffer_->state.store(BufferState::ReadyForRead);

    return;
  }
};

}  // namespace HugeCTR

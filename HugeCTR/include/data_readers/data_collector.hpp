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

#include <unistd.h>

#include <atomic>
#include <common.hpp>
#include <memory>
#include <mutex>
#include <queue>
#include <resource_manager.hpp>
#include <thread>
#include <utils.hpp>

#include "data_readers/data_reader_common.hpp"
#ifdef ENABLE_MPI
#include <mpi.h>
#endif
namespace HugeCTR {

template <typename TypeComp>
void split(Tensor2<float>& label_tensor, Tensor2<TypeComp>& dense_tensor,
           const Tensor2<float>& label_dense_buffer, const int label_dense_dim,
           cudaStream_t stream);

/**
 * @brief A helper class of data reader.
 *
 * This class implement asynchronized data collecting from heap
 * to output of data reader, thus data collection and training
 * can work in a pipeline.
 */
template <typename T>
class DataCollector {
  class BackgroundDataCollectorThread {
    std::vector<std::shared_ptr<ThreadBuffer>> thread_buffers_;
    std::shared_ptr<BroadcastBuffer> broadcast_buffer_;

    std::atomic<bool> loop_flag_;
    int counter_;
    std::vector<size_t> last_batch_nnz_;  // local_gpu_count * embedding number
    std::vector<char> worker_status_;
    int eof_worker_num_;

    std::shared_ptr<ResourceManager> resource_manager_;

   public:
    BackgroundDataCollectorThread(const std::vector<std::shared_ptr<ThreadBuffer>> &thread_buffers,
                                  const std::shared_ptr<BroadcastBuffer> &broadcast_buffer,
                                  const std::shared_ptr<ResourceManager> &resource_manager)
        : thread_buffers_(thread_buffers),
          broadcast_buffer_(broadcast_buffer),
          loop_flag_{true},
          counter_{0},
          last_batch_nnz_(
              broadcast_buffer->is_fixed_length.size() * resource_manager->get_local_gpu_count(),
              0),
          worker_status_(thread_buffers.size(), 0),
          eof_worker_num_(0),
          resource_manager_(resource_manager) {}

    void start() {
      while (loop_flag_.load()) {
        auto &current_src_buffer = thread_buffers_[counter_];
        // auto &next_src_buffer = thread_buffers_[(counter_ + 1) % thread_buffers_.size()];
        auto &dst_buffer = broadcast_buffer_;
        auto src_expected = BufferState::ReadyForRead;
        auto dst_expected = BufferState::ReadyForWrite;
        int local_gpu_count = resource_manager_->get_local_gpu_count();
        int batch_size = current_src_buffer->batch_size;
        int label_dim = current_src_buffer->label_dim;
        int dense_dim = current_src_buffer->dense_dim;
        int param_num = current_src_buffer->param_num;

        int batch_size_per_gpu = batch_size / resource_manager_->get_global_gpu_count();
        
        if(worker_status_[counter_]) {
          counter_ = (counter_ + 1) % thread_buffers_.size();
          continue;
        }
        

        if ((current_src_buffer->state.load() == BufferState::Reading ||
             current_src_buffer->state.compare_exchange_weak(src_expected, BufferState::Reading)) && (dst_buffer->state.load() == BufferState::Writing ||
            dst_buffer->state.compare_exchange_weak(dst_expected, BufferState::Writing))){
            assert(current_src_buffer->state.load() == BufferState::Reading);
            assert(dst_buffer->state.load() == BufferState::Writing);

            if(current_src_buffer->current_batch_size == 0) {
              worker_status_[counter_] = 1;
              eof_worker_num_ += 1;
              current_src_buffer->state.store(BufferState::FileEOF);
            }
            if(static_cast<size_t>(eof_worker_num_) != thread_buffers_.size() && current_src_buffer->current_batch_size == 0) {
              counter_ = (counter_ + 1) % thread_buffers_.size();
              dst_buffer->state.store(BufferState::ReadyForWrite);
              continue;
            }
            dst_buffer->current_batch_size = current_src_buffer->current_batch_size;
            if(current_src_buffer->current_batch_size != 0) {
  #pragma omp parallel for num_threads(local_gpu_count)
              for (int i = 0; i < local_gpu_count; ++i) {
                auto local_gpu = resource_manager_->get_local_gpu(i);

                CudaDeviceContext ctx(local_gpu->get_device_id());

                for (int param_id = 0; param_id < param_num; ++param_id) {
                  auto src_sparse_tensor = SparseTensor<T>::stretch_from(
                      current_src_buffer->device_sparse_buffers[param_id]);
                  auto dst_sparse_tensor = SparseTensor<T>::stretch_from(
                      dst_buffer->sparse_buffers[i * param_num + param_id]);

                  if (current_src_buffer->is_fixed_length[param_id] &&
                      last_batch_nnz_[i * param_num + param_id] == src_sparse_tensor.nnz()) {
                    CK_CUDA_THROW_(cudaMemcpyAsync(
                        dst_sparse_tensor.get_value_ptr(), src_sparse_tensor.get_value_ptr(),
                        src_sparse_tensor.nnz() * sizeof(T), cudaMemcpyDeviceToDevice,
                        local_gpu->get_memcpy_stream()));
                  } else {
                    sparse_tensor_helper::cuda::copy_async(dst_sparse_tensor, src_sparse_tensor,
                                                          cudaMemcpyDeviceToDevice,
                                                          local_gpu->get_memcpy_stream());
                    last_batch_nnz_[i * param_num + param_id] = src_sparse_tensor.nnz();
                  }
                }

                auto dst_dense_tensor = Tensor2<float>::stretch_from(dst_buffer->dense_tensors[i]);
                auto src_dense_tensor =
                    Tensor2<float>::stretch_from(current_src_buffer->device_dense_buffers);
                CK_CUDA_THROW_(cudaMemcpyAsync(
                    dst_dense_tensor.get_ptr(),
                    src_dense_tensor.get_ptr() + i * batch_size_per_gpu * (label_dim + dense_dim),
                    batch_size_per_gpu * (label_dim + dense_dim) * sizeof(float),
                    cudaMemcpyDeviceToDevice, local_gpu->get_memcpy_stream()));
                CK_CUDA_THROW_(cudaStreamSynchronize(local_gpu->get_memcpy_stream()));
                // CK_CUDA_THROW_(cudaEventRecord(broadcast_buffer_->finish_broadcast_events[i],
                // local_gpu->get_memcpy_stream()));
                // CK_CUDA_THROW_(cudaEventSynchronize(broadcast_buffer_->finish_broadcast_events[i]));
              }
              current_src_buffer->state.store(BufferState::ReadyForWrite);
              counter_ = (counter_ + 1) % thread_buffers_.size();
            } else {
              memset(worker_status_.data(), 0, sizeof(char) * worker_status_.size());
              eof_worker_num_ = 0;
              counter_ = 0;
            }

            dst_buffer->state.store(BufferState::ReadyForRead);
        } else {
          usleep(2);
        }
      }
    }

    void stop() { loop_flag_.store(false); }
  };

  std::shared_ptr<BroadcastBuffer> broadcast_buffer_;
  std::shared_ptr<DataReaderOutput> output_buffer_;

  BackgroundDataCollectorThread background_collector_;
  std::thread background_collector_thread_;

  std::atomic<bool> loop_flag_;
  std::vector<size_t> last_batch_nnz_;

  std::shared_ptr<ResourceManager> resource_manager_;

 public:
  void stop(){ 
    background_collector_.stop();
  }
  DataCollector(const std::vector<std::shared_ptr<ThreadBuffer>> &thread_buffers,
                const std::shared_ptr<BroadcastBuffer> &broadcast_buffer,
                std::shared_ptr<DataReaderOutput> &output,
                const std::shared_ptr<ResourceManager> &resource_manager)
      : broadcast_buffer_(broadcast_buffer),
        output_buffer_(output),
        background_collector_(thread_buffers, broadcast_buffer, resource_manager),
        loop_flag_{true},
        last_batch_nnz_(
            broadcast_buffer->is_fixed_length.size() * resource_manager->get_local_gpu_count(), 0),
        resource_manager_(resource_manager) {
    background_collector_thread_ = std::thread([this]() { background_collector_.start(); });
  }

  ~DataCollector() {
    background_collector_.stop();
    background_collector_thread_.join();
  }

  long long read_a_batch_to_device() {
    // MESSAGE_("data collector waiting read_a_batch_to_device");
    BufferState expected = BufferState::ReadyForRead;
    while (!broadcast_buffer_->state.compare_exchange_weak(expected, BufferState::Reading)) {
      expected = BufferState::ReadyForRead;
      usleep(2);
    }
    long long current_batch_size = broadcast_buffer_->current_batch_size;
    if (current_batch_size != 0) {
      int local_gpu_count = resource_manager_->get_local_gpu_count();

#pragma omp parallel for num_threads(local_gpu_count)
      for (int i = 0; i < local_gpu_count; ++i) {
        auto local_gpu = resource_manager_->get_local_gpu(i);

        CudaDeviceContext ctx(local_gpu->get_device_id());

        // wait until last iteration finish
        CK_CUDA_THROW_(
            cudaEventRecord(output_buffer_->last_batch_finish_events[i], local_gpu->get_stream()));
        CK_CUDA_THROW_(cudaEventSynchronize(output_buffer_->last_batch_finish_events[i]));
        auto label_tensor = Tensor2<float>::stretch_from(output_buffer_->label_tensors[i]);
        auto label_dense_tensor = Tensor2<float>::stretch_from(broadcast_buffer_->dense_tensors[i]);

        for(size_t param_id = 0; param_id < output_buffer_->sparse_name_vec.size(); ++param_id) {
          const auto &top_name = output_buffer_->sparse_name_vec[param_id];
          int idx_broadcast = i * broadcast_buffer_->param_num + param_id;
          auto src_sparse_tensor =
              SparseTensor<T>::stretch_from(broadcast_buffer_->sparse_buffers[idx_broadcast]);
          if(output_buffer_->sparse_tensors_map.find(top_name) == output_buffer_->sparse_tensors_map.end()) {
            CK_THROW_(Error_t::IllegalCall, "can not find sparse name");
          }
          auto dst_sparse_tensor = SparseTensor<T>::stretch_from(output_buffer_->sparse_tensors_map[top_name][i]);

          if (broadcast_buffer_->is_fixed_length[idx_broadcast] &&
              last_batch_nnz_[idx_broadcast] ==
                  src_sparse_tensor.nnz()) {
            CK_CUDA_THROW_(cudaMemcpyAsync(dst_sparse_tensor.get_value_ptr(),
                                           src_sparse_tensor.get_value_ptr(),
                                           src_sparse_tensor.nnz() * sizeof(T),
                                           cudaMemcpyDeviceToDevice, local_gpu->get_stream()));
          } else {
            sparse_tensor_helper::cuda::copy_async(dst_sparse_tensor, src_sparse_tensor,
                                                   cudaMemcpyDeviceToDevice,
                                                   local_gpu->get_stream());
            last_batch_nnz_[idx_broadcast] = src_sparse_tensor.nnz();
          }
        }
        const int label_dense_dim = output_buffer_->label_dense_dim;

        if (output_buffer_->use_mixed_precision) {
          auto dense_tensor = Tensor2<__half>::stretch_from(output_buffer_->dense_tensors[i]);
          split(label_tensor, dense_tensor, label_dense_tensor, label_dense_dim,
                local_gpu->get_stream());
        } else {
          auto dense_tensor = Tensor2<float>::stretch_from(output_buffer_->dense_tensors[i]);
          split(label_tensor, dense_tensor, label_dense_tensor, label_dense_dim,
                local_gpu->get_stream());
        }
      }
    }else {
      broadcast_buffer_->state.store(BufferState::ReadyForWrite);
    }
    return current_batch_size;
  }

  void finalize_batch() {
    for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
      const auto &local_gpu = resource_manager_->get_local_gpu(i);
      CudaDeviceContext context(local_gpu->get_device_id());
      cudaStreamSynchronize(local_gpu->get_stream());
    }

    broadcast_buffer_->state.store(BufferState::ReadyForWrite);
  }
};
}  // namespace HugeC

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

#include <nvToolsExt.h>
#include <omp.h>

#include <atomic>
#include <common.hpp>
#include <data_readers/data_reader_common.hpp>
#include <memory>
#include <mutex>
#include <queue>
#include <resource_manager.hpp>
#include <thread>
#include <utils.hpp>

namespace HugeCTR {

namespace core23_reader {
template <typename TypeComp>
void split(Tensor2<float> &label_tensor, Tensor2<TypeComp> &dense_tensor,
           const core23::Tensor &label_dense_buffer, const int label_dense_dim,
           cudaStream_t stream);
template <typename T>
void broadcast(const std::shared_ptr<ThreadBuffer23> &thread_buffer,
               std::shared_ptr<BroadcastBuffer23> &broadcast_buffer,
               std::vector<size_t> &last_batch_nnz_,
               const std::shared_ptr<ResourceManager> &resource_manager);
template <typename T>
class DataCollector {
  class BackgroundDataCollectorThread {
    // TODO remove me
    std::vector<std::shared_ptr<ThreadBuffer23>> thread_buffers_;
    std::shared_ptr<BroadcastBuffer23> broadcast_buffer_;

    std::atomic<bool> loop_flag_;
    int counter_;
    std::vector<size_t> last_batch_nnz_;  // local_gpu_count * embedding number
    std::vector<char> worker_status_;
    int eof_worker_num_;

    std::shared_ptr<ResourceManager> resource_manager_;

   public:
    BackgroundDataCollectorThread(
        const std::vector<std::shared_ptr<ThreadBuffer23>> &thread_buffers,
        const std::shared_ptr<BroadcastBuffer23> &broadcast_buffer,
        const std::shared_ptr<ResourceManager> &resource_manager);
    void start();
    void stop();
  };
  std::shared_ptr<BroadcastBuffer23> broadcast_buffer_;
  std::shared_ptr<DataReaderOutput> output_buffer_;

  BackgroundDataCollectorThread background_collector_;
  std::thread background_collector_thread_;

  std::atomic<bool> loop_flag_;
  std::vector<size_t> last_batch_nnz_;

  std::shared_ptr<ResourceManager> resource_manager_;

 public:
  void stop();
  DataCollector(const std::vector<std::shared_ptr<ThreadBuffer23>> &thread_buffers,
                const std::shared_ptr<BroadcastBuffer23> &broadcast_buffer,
                std::shared_ptr<DataReaderOutput> &output,
                const std::shared_ptr<ResourceManager> &resource_manager);
  ~DataCollector();

  long long read_a_batch_to_device();

  void finalize_batch();
};
};  // namespace core23_reader
template <typename TypeComp>
void split(Tensor2<float> &label_tensor, Tensor2<TypeComp> &dense_tensor,
           const Tensor2<float> &label_dense_buffer, const int label_dense_dim,
           cudaStream_t stream);

template <typename T>
void broadcast(const std::shared_ptr<ThreadBuffer> &thread_buffer,
               std::shared_ptr<BroadcastBuffer> &broadcast_buffer,
               std::vector<size_t> &last_batch_nnz_,
               const std::shared_ptr<ResourceManager> &resource_manager);

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
                                  const std::shared_ptr<ResourceManager> &resource_manager);

    void start();

    void stop();
  };
  std::shared_ptr<BroadcastBuffer> broadcast_buffer_;
  std::shared_ptr<DataReaderOutput> output_buffer_;

  BackgroundDataCollectorThread background_collector_;
  std::thread background_collector_thread_;

  std::atomic<bool> loop_flag_;
  std::vector<size_t> last_batch_nnz_;

  std::shared_ptr<ResourceManager> resource_manager_;

 public:
  void stop();
  DataCollector(const std::vector<std::shared_ptr<ThreadBuffer>> &thread_buffers,
                const std::shared_ptr<BroadcastBuffer> &broadcast_buffer,
                std::shared_ptr<DataReaderOutput> &output,
                const std::shared_ptr<ResourceManager> &resource_manager);
  ~DataCollector();

  long long read_a_batch_to_device();
  void finalize_batch();
};
}  // namespace HugeCTR

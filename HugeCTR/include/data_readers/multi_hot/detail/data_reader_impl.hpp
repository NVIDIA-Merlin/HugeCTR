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

#include <cstddef>
#include <data_readers/multi_hot/detail/atomic_wrapper.hpp>
#include <data_readers/multi_hot/detail/batch_file_reader.hpp>
#include <data_readers/multi_hot/detail/device_transfer.hpp>
#include <data_readers/multi_hot/detail/system_latch.hpp>
#include <future>
#include <memory>
#include <optional>
#include <resource_manager.hpp>
#include <thread>
#include <unordered_map>
#include <vector>

namespace HugeCTR {
namespace MultiHot {

//#define BENCH_IO

struct FileSource {
  std::string name;
  size_t sample_size_bytes;
  size_t slot_id;
};

enum BatchState {
  NOT_READY = 0,
  READY_TO_UPLOAD,
  READY_TO_CONSUME,
};

class DataReaderImpl {
 public:
  class Batch {
    friend class DataReaderImpl;

   public:
    uint8_t* get_device_data(size_t device_id, size_t slot) const {
      assert(device_id < local_batches.size());
      size_t idx = 0;
      for (auto io_batch : local_batches[device_id].io_batches) {
        if (io_batch->slot_id == slot) {
          return local_batches[device_id].device_data[idx];
        }
        idx++;
      }
      throw std::runtime_error("get_device_data() called with invalid slot " +
                               std::to_string(slot));
      return nullptr;
    }

    uint8_t* get_host_data(size_t device_id, size_t slot) const {
      assert(device_id < local_batches.size());
      for (auto io_batch : local_batches[device_id].io_batches) {
        if (io_batch->slot_id == slot) {
          return io_batch->data;
        }
      }
      throw std::runtime_error("get_host_data() called with invalid slot " + std::to_string(slot));
      return nullptr;
    }

    size_t get_id() const { return id; }

    size_t get_local_batch_size_bytes(size_t device_id, size_t slot) const {
      assert(device_id < local_batches.size());
      assert(slot < local_batches[device_id].io_batches.size());
      return local_batches[device_id].io_batches[slot]->shard_size_bytes;
    }

    size_t get_batch_size_bytes() const { return local_batches[0].io_batches[0]->batch_size_bytes; }

   private:
    struct LocalBatch {
      std::vector<const BatchFileReader::Batch*> io_batches;  // [slot]
      // For DP, device_data.size() == 1, for MP, device_data.size() == num_features
      std::vector<uint8_t*> device_data;
      std::vector<DeviceTransfer*> device_transfers;  // max slots in length
      AtomicWrapper<size_t> num_transfers;
    };

    std::atomic<BatchState> state;
    std::vector<LocalBatch> local_batches;
    size_t id;
    size_t total_ios;
    std::atomic<size_t> num_completed_io;
    std::atomic<size_t> num_completed_uploads;
    std::atomic<size_t> in_use_count;
  };

  /**
   * @param data_files Files for this node. E.g {["input0.bin", {0}],
   *                                             ["input1.bin", {1}],
   *                                             ["input2.bin", {2,3}],
   *                                             ["common.bin", {0,1,2,3}]}
   * @param batch_size Number of samples per batch
   * @param num_threads_per_file Number of threads per file reader
   * @param num_batches_per_thread Number of inflight batches per thread reader
   * @param gpus_slot_ownership_matrix Matrix indicating what GPUs own which slots. E.g:
   *                                Node 0: GPU | 0 1 2 3       Node 1: GPU | 0 1 2 3
   *                                        -------------               -------------
   *                                 common.bin | 1 1 1 1        common.bin | 1 1 1 1
   *                                input_0.bin | 1 1 0 0       input_0.bin | 0 0 0 0
   *                                input_1.bin | 0 1 0 0       input_1.bin | 0 0 0 0
   *                                input_2.bin | 0 0 1 1       input_2.bin | 0 0 0 1
   *                                input_3.bin | 1 0 0 0       input_3.bin | 0 0 0 0
   *                                input_4.bin | 0 0 0 1       input_4.bin | 0 0 0 0
   *                                input_5.bin | 0 0 0 0       input_5.bin | 1 0 0 0
   *                                input_6.bin | 0 0 0 0       input_6.bin | 0 1 0 0
   *                                input_7.bin | 0 0 0 0       input_7.bin | 0 0 0 1
   */
  DataReaderImpl(const std::vector<FileSource>& source_files,
                 const std::shared_ptr<ResourceManager>& resource_manager, size_t batch_size,
                 size_t num_threads_per_file, size_t num_batches_per_thread, bool shuffle,
                 bool schedule_uploads);
  ~DataReaderImpl();

  void start();

  const Batch& get_batch();

  void device_release_last_batch_here(cudaStream_t stream) const;

  void schedule_upload_here(int device_id, cudaStream_t stream, bool from_graph);

  void upload_notify(int device_id);

  size_t get_total_inflight_batches() const;

 private:
  Batch& get_parent(size_t batch_i);

  void read_batches(BatchFileReader& file_reader, int device_id);

  void upload_batches(size_t device_id);

  std::unique_ptr<IBatchLocations> configure_locations(FileSource source, size_t batch_size,
                                                       bool shuffle) const;

  static void CUDART_CB release_batch_callback(cudaStream_t stream, cudaError_t status,
                                               void* user_data);

  void compute_batch_stats(Batch* batch);

  struct IOStats {
    double batch_min_latency;
    double batch_max_latency;
    double batch_avg_latency;
  } io_stats;

  std::shared_ptr<ResourceManager> resource_manager_;
  size_t batch_i_ = 0;
  size_t num_batches_ = 0;
  volatile bool running_ = false;
  bool schedule_uploads_ = false;
  Batch* last_batch_ = nullptr;
  std::vector<std::unique_ptr<Batch>> batch_buffers_;

  std::unordered_map<int, std::vector<std::unique_ptr<BatchFileReader>>> file_readers_;
  std::vector<std::thread> file_reader_threads_;

  std::vector<std::thread> placement_threads_;
  std::vector<AtomicWrapper<size_t>> pending_transfers_;
  std::vector<cudaStream_t> placement_streams_;
  std::vector<cudaEvent_t> placement_events_;
};

}  // namespace MultiHot
}  // namespace HugeCTR
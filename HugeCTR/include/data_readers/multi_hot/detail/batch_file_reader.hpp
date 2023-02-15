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

#include <data_readers/multi_hot/detail/batch_locations.hpp>
#include <data_readers/multi_hot/detail/io_context.hpp>
#include <data_readers/multi_hot/detail/time_helper.hpp>
#include <data_readers/multi_hot/detail/work_queue.hpp>
#include <memory>
#include <queue>

// High level
// - specify num batches inflight
// - cached in host memory (128)

// s0 nnz0 nnz1 nnz2 0 1 2 3 4 5 6 7 8 9
// s1 nnz0 nnz1 nnz2 0 1 2 3 4 5 6 7 8 9
// [a b c d e f]
// nnz: [2, 1, 3]

// 1. If the batch is sharded across multiple files, how do I enforce ordering?
//      BatchLocations iterator needs to provide same iterator across each file

// Requirements, batch can be distributed across multiple files
// Requirements, need to be able to enforce ordering from io across multiple files
// Requirements, on node 0, only need slots 1 & 2, only need to read feature-major files 1 & 2.

// Each batch file reader names the same IDs, but can be mapped to different locations

namespace HugeCTR {
class BatchFileReader {
 public:
  class Batch {
    friend class BatchFileReader;

   public:
    Batch(BatchFileReader* __reader, uint8_t* __data, size_t __size_bytes, size_t __slot)
        : data(nullptr),
          size_bytes(__size_bytes),
          slot_id(__slot),
          batch_id(0),
          batch_i(0),
          aligned_data(__data),
          reader(__reader) {}

    void release() { reader->release_batch(this); }

    uint8_t* data = nullptr;
    size_t size_bytes = 0;
    size_t slot_id = 0;
    size_t batch_id = 0;
    size_t batch_i = 0;
    double start_time = 0.f;
    double end_time = 0.f;

   private:
    uint8_t* aligned_data = nullptr;
    BatchFileReader* reader;
  };

  BatchFileReader(const std::string& fname, size_t slot, size_t max_batches_inflight,
                  std::unique_ptr<IBatchLocations> batch_locations);
  BatchFileReader(const BatchFileReader& other) = delete;
  ~BatchFileReader();

  // max_batches_inflight batches can be returned in any order
  const std::vector<const Batch*>& read_batches(size_t timeout_us = 10);
  void release_batch(const Batch* batch);
  size_t get_queue_depth() const;

 private:
  void submit_reads();
  const std::vector<const Batch*>& collect(size_t timeout_us);

  // Files stored feature-major (i.e multi-hot) will have a distinct slot_id for each file
  // Files that are batch-major will have the same slot_id value of 0.
  size_t slot_id_ = 0;
  size_t max_batches_inflight_ = 0;

  std::vector<const Batch*> tmp_completed_batches_;
  // To handle the case where a batch doesn't span all numas but we still need to return a read
  // request.
  std::vector<const Batch*> empty_batches_;
  std::vector<Batch> batches_;      // So we can free memory if not all batches are released
  WorkQueue<Batch*> free_batches_;  // TODO: Can be optimized to SPSC queue instead of MPMC

  std::unique_ptr<IBatchLocations> batch_locations_;
  IBatchLocations::iterator batch_locations_iterator_;
  std::unique_ptr<IOContext> io_ctx_;

  int fd_;
  size_t buf_size_ = 0;  // used for numa_free
  std::atomic<size_t> num_inflight_ = {0};
};
}  // namespace HugeCTR
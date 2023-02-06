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

#include <algorithm>
#include <atomic>
#include <common.hpp>
#include <fstream>
#include <random>
#include <vector>

namespace HugeCTR {

struct MmapOffset {
  char* offset;
  long long samples;
};

/**
 * @brief A threads safe file list implementation.
 *
 * FileList reads file list from text file, and maintains a vector of file name. It supports
 * getting file names with multiple threads. All the threads will get the names in order.
 * Text file begins with the number of files, and then the list of file names.
 * @verbatim
 * Text file example:
 * 3
 * 1.txt
 * 2.txt
 * 3.txt
 * @endverbatim
 */
class MmapOffsetList {
 private:
  const long long length_;
  std::vector<MmapOffset> offsets_;
  std::atomic<long long> counter_{0};
  const int num_workers_;
  bool repeat_;
  char* mmapped_data_;
  int fd_;

 public:
  // stride: samle size in byte
  MmapOffsetList(const std::string& file_name, long long num_samples, long long stride,
                 long long batchsize, bool use_shuffle, int num_workers, bool repeat);

  ~MmapOffsetList();

  MmapOffset get_offset(long long round, int worker_id) {
    size_t worker_pos = round * num_workers_ + worker_id;
    if (!repeat_ && worker_pos >= offsets_.size()) {
      throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
    }
    size_t counter = (round * num_workers_ + worker_id) % offsets_.size();
    if (worker_id >= num_workers_) {
      HCTR_OWN_THROW(Error_t::WrongInput, "worker_id >= num_workers_");
    }
    if (counter == offsets_.size() - 1) {
      // HCTR_OWN_THROW(Error_t::OutOfBound, "End of File");
      HCTR_LOG_S(INFO, WORLD) << "End of File, worker:  " << worker_id << std::endl;
    }
    return offsets_[counter];
  }
};

}  // namespace HugeCTR

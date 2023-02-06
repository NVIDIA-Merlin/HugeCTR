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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <data_readers/mmap_offset_list.hpp>

namespace HugeCTR {

MmapOffsetList::MmapOffsetList(const std::string& file_name, long long num_samples,
                               long long stride, long long batchsize, bool use_shuffle,
                               int num_workers, bool repeat)
    : length_(num_samples * stride), num_workers_(num_workers), repeat_(repeat) {
  try {
    fd_ = open(file_name.c_str(), O_RDONLY, 0);
    if (fd_ == -1) {
      HCTR_OWN_THROW(Error_t::BrokenFile, "Error open file for read");
      return;
    }

    /* Get the size of the file. */
    mmapped_data_ = (char*)mmap(0, length_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmapped_data_ == MAP_FAILED) {
      close(fd_);
      HCTR_OWN_THROW(Error_t::BrokenFile, "Error mmapping the file");
      return;
    }

    auto offset_gen = [stride](char* mmapped_data, long long idx, long long samples) -> MmapOffset {
      char* offset = mmapped_data + idx * stride;
      return {offset, samples};
    };

    offsets_.reserve(num_samples);
    for (long long sample_idx = 0; sample_idx < num_samples; sample_idx += batchsize) {
      if (sample_idx + batchsize <= num_samples) {
        offsets_.emplace_back(offset_gen(mmapped_data_, sample_idx, batchsize));
      } else {
        offsets_.emplace_back(offset_gen(mmapped_data_, sample_idx, num_samples - sample_idx));
      }
    }
    // shuffle
    if (use_shuffle) {
      std::random_device rd;
      unsigned int seed = rd();

#ifdef ENABLE_MPI
      HCTR_MPI_THROW(MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
#endif
      auto rng = std::default_random_engine{seed};
      std::shuffle(std::begin(offsets_), std::end(offsets_), rng);
    }

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

MmapOffsetList::~MmapOffsetList() {
  munmap(mmapped_data_, length_);
  close(fd_);
}

}  // namespace HugeCTR
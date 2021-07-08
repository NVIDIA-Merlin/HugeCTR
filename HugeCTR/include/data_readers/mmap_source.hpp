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

#include <data_readers/mmap_offset_list.hpp>
#include <data_readers/source.hpp>

namespace HugeCTR {
class MmapSource : public Source {
 private:
  std::shared_ptr<MmapOffsetList> mmap_offset_list_;
  MmapOffset offset_;
  int worker_id_;
  long long round_{0};

 public:
  MmapSource(std::shared_ptr<MmapOffsetList> mmap_offset_list, int worker_id)
      : mmap_offset_list_(mmap_offset_list), worker_id_(worker_id) {}

  char* get_ptr() { return offset_.offset; }

  // no use here
  bool is_open() noexcept { return true; }

  Error_t next_source() noexcept {
    try {
      offset_ = mmap_offset_list_->get_offset(round_, worker_id_);
      round_++;
      return Error_t::Success;
    } catch (const internal_runtime_error& rt_err) {
      Error_t err = rt_err.get_error();
      if (err == Error_t::EndOfFile) {
        return Error_t::EndOfFile;
      } else {
        return Error_t::UnspecificError;
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      return Error_t::UnspecificError;
    }
  }

  long long get_num_of_items_in_source() { return offset_.samples; }
};
}  // namespace HugeCTR

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

#include <errno.h>
#include <sys/types.h>
#include <unistd.h>

#include <data_readers/raw_offset_list.hpp>
#include <data_readers/source.hpp>

namespace HugeCTR {
class RawSource : public Source {
 private:
  std::shared_ptr<RawOffsetList> raw_offset_list_;
  FileOffset offset_;
  int worker_id_;
  long long round_{0};
  int fd_;
  char* buffer_;
  long long batch_size_;
  long long stride_;  // bytes per batch
  const size_t alignment_bytes_ = 512;

 public:
  RawSource(std::shared_ptr<RawOffsetList> raw_offset_list, int worker_id)
      : raw_offset_list_(raw_offset_list), worker_id_(worker_id) {
    fd_ = open(raw_offset_list_->get_file_name().c_str(), O_RDONLY | O_DIRECT);
    if (fd_ == -1) {
      if (errno == EINVAL) {
        // try without O_DIRECT flag
        fd_ = open(raw_offset_list_->get_file_name().c_str(), O_RDONLY);
        if (fd_ == -1) {
          std::cout << "File open error: " << strerror(errno) << std::endl;
          CK_THROW_(Error_t::BrokenFile, "Error open file for read");
        }
      } else {
        CK_THROW_(Error_t::BrokenFile, "Error open file for read");
      }
    }
    batch_size_ = raw_offset_list_->get_batch_size();
    stride_ = raw_offset_list_->get_stride();
    buffer_ = static_cast<char*>(
        aligned_alloc(alignment_bytes_, batch_size_ * stride_ + alignment_bytes_));
  }

  ~RawSource() {
    free(buffer_);
    close(fd_);
  }

  char* get_ptr() {
    size_t req_beg_offset = (size_t)offset_.offset;
    size_t req_end_offset = req_beg_offset + (size_t)offset_.samples * stride_;
    size_t raw_beg_offset = (req_beg_offset / alignment_bytes_) * alignment_bytes_;
    size_t raw_end_offset =
        ((req_end_offset + alignment_bytes_ - 1) / alignment_bytes_) * alignment_bytes_;

    if (lseek(fd_, raw_beg_offset, SEEK_SET) == -1) {
      CK_THROW_(Error_t::BrokenFile, "File seek read");
    }
    if (::read(fd_, buffer_, raw_end_offset - raw_beg_offset) < 0) {
      CK_THROW_(Error_t::BrokenFile, "File read failed");
    }
    return buffer_ + (req_beg_offset - raw_beg_offset);
  }

  // no use here
  bool is_open() noexcept { return true; }

  Error_t next_source() noexcept {
    try {
      offset_ = raw_offset_list_->get_offset(round_, worker_id_);
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

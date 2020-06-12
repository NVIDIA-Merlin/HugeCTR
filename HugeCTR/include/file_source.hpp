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
#include <fstream>
#include <vector>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/file_list.hpp"
#include "HugeCTR/include/source.hpp"

namespace HugeCTR {

class FileSource : public Source {
 private:
  FileList file_list_;          /**< file list of data set */
  std::ifstream in_file_stream_; /**< file stream of data set file */
  std::string file_name_;        /**< file name of current file */
  const long long offset_;
  const long long stride_;
  unsigned int counter_{0};
 public:
  FileSource(long long offset, long long stride, const std::string& file_list)
    :  file_list_(file_list), offset_(offset), stride_(stride) {}

  /**
   * Read "bytes_to_read" byte to the memory associated to ptr.
   * @param ptr pointer to user located buffer
   * @param bytes_to_read bytes to read
   * @return `FileCannotOpen` `OutOfBound` `Success` `UnspecificError`
   */
  Error_t read(char* ptr, size_t bytes_to_read) noexcept {
    try {
      if (!in_file_stream_.is_open()) {
        return Error_t::FileCannotOpen;
      }
      if (bytes_to_read > 0) {
        in_file_stream_.read(ptr, bytes_to_read);
      }
      if (in_file_stream_.gcount() != static_cast<int>(bytes_to_read)) {
        return Error_t::OutOfBound;
      }
      return Error_t::Success;
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      return Error_t::UnspecificError;
    }
  }

  /**
   * Start a new file to read.
   * @return `Success`, `FileCannotOpen` or `UnspecificError`
   */
  Error_t next_source() noexcept {
    try {
      if (in_file_stream_.is_open()) {
        in_file_stream_.close();
      }
      std::string file_name = file_list_.get_a_file_with_id(offset_ + counter_ * stride_);
      counter_++;  // counter_ should be accum for every source.
      in_file_stream_.open(file_name, std::ifstream::binary);
      if (!in_file_stream_.is_open()) {
        CK_RETURN_(Error_t::FileCannotOpen, "in_file_stream_.is_open() failed: " + file_name);
      }
      return Error_t::Success;
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      return Error_t::UnspecificError;
    }
  }

  bool is_open() noexcept { return in_file_stream_.is_open(); }
};

}  // namespace HugeCTR

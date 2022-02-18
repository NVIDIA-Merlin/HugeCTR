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

namespace HugeCTR {

class Source {
 public:
  virtual ~Source() = default;

  /**
   * Read "bytes_to_read" byte to the memory associated to ptr.
   * @param ptr pointer to user located buffer
   * @param bytes_to_read bytes to read
   * @return `DataCheckError` `OutOfBound` `Success` `UnspecificError`
   */
  virtual Error_t read(char* ptr, size_t bytes_to_read) {
    HCTR_OWN_THROW(Error_t::BrokenFile, "Invalid Call");
    return Error_t::Success;
  }

  virtual char* get_ptr() {
    HCTR_OWN_THROW(Error_t::BrokenFile, "Invalid Call");
    return nullptr;
  }

  /**
   * Start a new file to read.
   * @return `FileCannotOpen` or `UnspecificError`
   */
  virtual Error_t next_source() noexcept = 0;

  virtual long long get_num_of_items_in_source() { return 0; }

  virtual bool is_open() noexcept = 0;
};

}  // namespace HugeCTR

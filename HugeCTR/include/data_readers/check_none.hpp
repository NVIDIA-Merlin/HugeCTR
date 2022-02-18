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

#include <data_readers/checker.hpp>
#include <data_readers/source.hpp>

namespace HugeCTR {

class CheckNone : public Checker {
 private:
  const int MAX_TRY{10};

 public:
  CheckNone(Source& src) : Checker(src) {}
  /**
   * Read "bytes_to_read" byte to the memory associated to ptr.
   * Users don't need to manualy maintain the check bit offset, just specify
   * number of bytes you really want to see in ptr.
   * @param ptr pointer to user located buffer
   * @param bytes_to_read bytes to read
   * @return `DataCheckError` `OutOfBound` `Success` `UnspecificError`
   */
  Error_t read(char* ptr, size_t bytes_to_read) noexcept {
    try {
      Checker::src_.read(ptr, bytes_to_read);
      return Error_t::Success;
    } catch (const std::runtime_error& rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
      return Error_t::BrokenFile;
    }
  }

  /**
   * Start a new file to read.
   * @return `FileCannotOpen` or `UnspecificError`
   */
  Error_t next_source() {
    for (int i = MAX_TRY; i > 0; i--) {
      Error_t flag_eof = Checker::src_.next_source();
      if (flag_eof == Error_t::Success || flag_eof == Error_t::EndOfFile) {
        return flag_eof;
      }
    }
    HCTR_OWN_THROW(Error_t::FileCannotOpen,
                   "Checker::src_.next_source() == Error_t::Success failed");
    return Error_t::FileCannotOpen;  // to elimate compile error
  }
};

}  // namespace HugeCTR

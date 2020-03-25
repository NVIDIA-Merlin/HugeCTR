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

#include "HugeCTR/include/source.hpp"

namespace HugeCTR {

class Checker {
protected:
  Source& src_;
public:
  Checker(Source& src): src_(src){}
  /**
   * Read "bytes_to_read" byte to the memory associated to ptr.
   * Users don't need to manualy maintain the check bit offset, just specify
   * number of bytes you really want to see in ptr.
   * @param ptr pointer to user located buffer
   * @param bytes_to_read bytes to read
   * @return `DataCheckError` `OutOfBound` `Success` `UnspecificError`
   */
  virtual Error_t read(char* ptr, size_t bytes_to_read) noexcept = 0;

  /**
   * Start a new file to read.
   * @return `FileCannotOpen` or `UnspecificError`
   */
  virtual void next_source() = 0;

  virtual bool is_open() noexcept{
    return src_.is_open();
  }

  unsigned int get_source_counter(){
    return src_.get_counter();
  }

};

} //namespace HugeCTR

/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

namespace HugeCTR {

class Source {
public:
  /**
   * Read "bytes_to_read" byte to the memory associated to ptr.
   * @param ptr pointer to user located buffer
   * @param bytes_to_read bytes to read
   * @return `DataCheckError` `OutOfBound` `Success` `UnspecificError`
   */
  virtual Error_t read(char* ptr, size_t bytes_to_read) noexcept = 0;
  
  /**
   * Start a new file to read.
   * @return `FileCannotOpen` or `UnspecificError`
   */
  virtual Error_t next_source() noexcept = 0;

};

} //namespace HugeCTR

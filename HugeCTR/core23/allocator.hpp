/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <core23/cuda_stream.hpp>
#include <cstdint>

namespace HugeCTR {

namespace core23 {

class Allocator {
 public:
  virtual ~Allocator() {}

  [[nodiscard]] virtual void* allocate(int64_t size, CUDAStream stream = CUDAStream()) = 0;

  virtual void deallocate(void* ptr, CUDAStream stream = CUDAStream()) = 0;

  virtual int64_t default_alignment() const = 0;
  int64_t get_valid_alignment(int64_t alignment) const {
    return (alignment != 0 &&
            (default_alignment() % alignment == 0 || alignment % default_alignment() != 0))
               ? default_alignment()
               : alignment;
  }
};

}  // namespace core23

}  // namespace HugeCTR

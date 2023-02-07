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

struct BufferRequirements {
  int64_t num_bytes;
  int64_t alignment;
  CUDAStream stream;
};

}  // namespace core23
}  // namespace HugeCTR

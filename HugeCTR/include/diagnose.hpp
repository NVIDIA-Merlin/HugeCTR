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
#include <tensor2.hpp>

namespace HugeCTR {

namespace diagnose {

template <typename T>
void verify_and_histogram(const char* category, const Tensor2<T>& tensor,
                          const cudaStream_t& stream);

template <typename T>
void sample_and_print(const char* category, const Tensor2<T>& tensor, size_t sample_count,
                      const cudaStream_t& stream);

template <typename T>
void sample_and_print(const char* category, const Tensor2<T>& tensor, int begin, int end,
                      const cudaStream_t& stream);

template <typename T>
void dump(const char* filename, const Tensor2<T>& tensor, const cudaStream_t& stream);

}  // namespace diagnose

}  // namespace HugeCTR

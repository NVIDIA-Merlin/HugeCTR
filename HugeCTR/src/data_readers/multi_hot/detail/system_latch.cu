/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <data_readers/multi_hot/detail/system_latch.hpp>

namespace HugeCTR {

template <typename T>
__global__ void kernel_count_down(T* latch, T n) {
  atomicDec_system(latch, n);
}

void SystemLatch::device_count_down(cudaStream_t stream, value_type n, bool from_graph) {
  if (from_graph) {
    kernel_count_down<<<1, 1, 0, stream>>>(latch_, n);
  } else {
    cudaStreamAddCallback(stream, callback, (void*)latch_, 0);
  }
}

}  // namespace HugeCTR
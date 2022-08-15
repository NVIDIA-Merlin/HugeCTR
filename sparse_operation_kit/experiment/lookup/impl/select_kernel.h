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

#ifndef SELECT_KERNEL_H
#define SELECT_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace sok {

template <typename KeyType>
class SelectLauncher {
 public:
  void initialize(size_t num_splits);
  void operator()(const void* indices, size_t num_keys, void* output, void* output_buffer,
                  void* order, void* order_buffer, void* splits, size_t num_splits,
                  cudaStream_t stream = 0);

 private:
  int sm_count_;
  int warp_size_;
  int max_shared_memory_per_sm_;
  size_t key_warps_per_block_;
  size_t items_per_gpu_per_warp_;
  std::vector<int32_t> host_splits_;
};

}  // namespace sok

#endif

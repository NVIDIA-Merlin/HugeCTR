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

#ifndef HOTNESS_KERNEL_H
#define HOTNESS_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace sok {

template <typename DType>
class HotnessCalLauncher {
 public:
  void initialize();
  void operator()(const void* row_length_recv_buffer, size_t local_batchsize, int num_lookup,
                  int num_gpus, void* output_device, void* output_host, cudaStream_t stream = 0);

 private:
  int sm_count_;
};

}  // namespace sok

#endif

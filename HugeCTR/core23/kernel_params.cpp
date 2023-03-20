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

#include <core23/kernel_params.hpp>

namespace HugeCTR {

namespace core23 {

KernelParams &KernelParams::init() {
  // FIX: do we don't run 3G embedding in hybridcards?(if a node have A100 and H100 at once)
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  this->num_sms = device_prop.multiProcessorCount;
  this->warp_size = device_prop.warpSize;
  this->max_thread_per_sm = device_prop.maxThreadsPerMultiProcessor;
  this->max_thread_per_block = device_prop.maxThreadsPerBlock;
  return *this;
}

}  // namespace core23
}  // namespace HugeCTR

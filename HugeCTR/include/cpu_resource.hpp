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
#include <ctpl/ctpl_stl.h>
#include <curand.h>

namespace HugeCTR {

class CPUResource {
  curandGenerator_t curand_generator_;
  std::shared_ptr<ctpl::thread_pool> thread_pool_; /**< cpu thread pool for training */

 public:
  CPUResource(unsigned long long seed, size_t thread_num);
  CPUResource(const CPUResource&) = delete;
  CPUResource& operator=(const CPUResource&) = delete;
  ~CPUResource();

  const curandGenerator_t& get_curand_generator() const { return curand_generator_; }
  const std::shared_ptr<ctpl::thread_pool>& get_thread_pool() { return thread_pool_; }
};
}  // namespace HugeCTR
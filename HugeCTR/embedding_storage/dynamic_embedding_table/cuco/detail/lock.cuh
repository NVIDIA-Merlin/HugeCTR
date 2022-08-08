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

#include <cuda/std/atomic>

namespace cuco {
namespace detail {}

template <cuda::thread_scope Scope>
class lock {
  mutable cuda::atomic<int, Scope> _lock;

 public:
  __device__ lock() : _lock{0} {}

  template <typename CG>
  __device__ void acquire(CG const &g, unsigned long long lane) const {
    if (g.thread_rank() == lane) {
      int expected = 1;
      while (!_lock.compare_exchange_weak(expected, 2, cuda::std::memory_order_acquire)) {
        expected = 1;
      }
    }
    g.sync();
  }

  template <typename CG>
  __device__ void release(CG const &g, unsigned long long lane) const {
    if (g.thread_rank() == lane) {
      _lock.store(1, cuda::std::memory_order_release);
    }
  }
};

}  // namespace cuco
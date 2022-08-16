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

namespace cuco {
namespace detail {

/**
 * @brief Rounds `v` to the nearest power of 2 greater than or equal to `v`.
 *
 * @param v
 * @return The nearest power of 2 greater than or equal to `v`.
 */
constexpr size_t next_pow2(size_t v) noexcept {
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return ++v;
}

template <typename CG, typename Element>
__device__ __forceinline__ void copy_array(CG const &g, uint32_t n, Element *t, Element const *s) {
  for (auto i = g.thread_rank(); i < n; i += g.size()) {
    t[i] = s[i];
  }
}

template <typename CG, typename Element, typename Initializer>
__device__ __forceinline__ void init_array(CG const &g, uint32_t n, Element *t,
                                           Initializer initializer) {
  for (auto i = g.thread_rank(); i < n; i += g.size()) {
    t[i] = initializer();
  }
}

template <typename CG, typename Element, typename Initializer>
__device__ __forceinline__ void init_and_copy_array(CG const &g, uint32_t n, Element *t1,
                                                    Element *t2, Initializer initializer) {
  for (auto i = g.thread_rank(); i < n; i += g.size()) {
    t1[i] = t2[i] = initializer();
  }
}

template <typename CG, typename Element>
__device__ __forceinline__ void accumulate_array(CG const &g, uint32_t n, Element *t,
                                                 Element const *u) {
  for (auto i = g.thread_rank(); i < n; i += g.size()) {
    t[i] += u[i];
  }
}

}  // namespace detail
}  // namespace cuco
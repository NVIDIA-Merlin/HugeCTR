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
 * @brief Gives value to use as alignment for a pair type that is at least the
 * size of the sum of the size of the first type and second type, or 16,
 * whichever is smaller.
 */
template <typename First, typename Second>
constexpr size_t pair_alignment() {
  return std::min(size_t{16}, next_pow2(sizeof(First) + sizeof(Second)));
}
}  // namespace detail

/**
 * @brief Custom pair type
 *
 * This is necessary because `thrust::pair` is under aligned.
 *
 * @tparam First
 * @tparam Second
 */
template <typename First, typename Second>
struct alignas(detail::pair_alignment<First, Second>()) pair {
  using first_type = First;
  using second_type = Second;

  pair() = default;
  ~pair() = default;
  pair(pair const &) = default;
  pair(pair &&) = default;
  pair &operator=(pair const &) = default;
  pair &operator=(pair &&) = default;

  __host__ __device__ constexpr pair(First const &f, Second const &s) : first{f}, second{s} {}

  template <typename F, typename S>
  __host__ __device__ constexpr pair(pair<F, S> const &p) : first{p.first}, second{p.second} {}

  First first;
  Second second;
};

template <typename F, typename S>
using pair_type = cuco::pair<F, S>;

}  // namespace cuco

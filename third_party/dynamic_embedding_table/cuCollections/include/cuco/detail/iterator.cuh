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
namespace detail {}  // namespace detail

template <typename pair_atomic_type, typename element_type>
class iterator {
  pair_atomic_type *slots_;
  element_type *elements_;
  uint32_t index_;

 public:
  __device__ iterator(pair_atomic_type *slots, element_type *elements, uint32_t index) noexcept
      : slots_(slots), elements_(elements), index_{index} {}

  __device__ uint32_t index() const { return index_; }

  __device__ typename pair_atomic_type::first_type &key() { return slots_->first; }

  __device__ typename pair_atomic_type::second_type &lock() const { return slots_->second; }

  __device__ element_type *value() const { return elements_; }

  __device__ bool operator!=(const iterator &other) const { return index_ != other.index_; }
};

template <typename pair_atomic_type, typename element_type>
class const_iterator {
  pair_atomic_type *slots_;
  element_type *elements_;
  uint32_t index_;

 public:
  __device__ const_iterator(pair_atomic_type *slots, element_type *elements,
                            uint32_t index) noexcept
      : slots_(slots), elements_(elements), index_{index} {}

  __device__ uint32_t index() const { return index_; }

  __device__ typename pair_atomic_type::first_type const &key() const { return slots_->first; }

  __device__ typename pair_atomic_type::second_type &lock() const { return slots_->second; }

  __device__ element_type const *value() const { return elements_; }

  __device__ bool operator!=(const const_iterator &other) const { return index_ != other.index_; }
};

}  // namespace cuco
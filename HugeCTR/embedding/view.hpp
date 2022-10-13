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
#include <cuda_runtime.h>
#include <inttypes.h>

#include "core/macro.hpp"

namespace embedding {

template <typename T>
HOST_DEVICE_INLINE int binary_search_index_lower_bound(const T *const arr, int num, T target) {
  int start = 0;
  int end = num;
  while (start < end) {
    int middle = (start + end) / 2;
    T value = arr[middle];
    if (value <= target) {
      start = middle + 1;
    } else {
      end = middle;
    }
  }
  return start - 1;
}

HOST_DEVICE_INLINE bool binary_search_index(const int *arr, int num, int target) {
  int start = 0;
  int end = num;
  while (start < end) {
    int middle = (start + end) / 2;
    int value = arr[middle];
    if (value == target) {
      return true;
    }
    if (value <= target) {
      start = middle + 1;
    } else {
      end = middle;
    }
  }
  return false;
}

template <typename T>
class ArrayView {
 public:
  using value_type = T;
  using size_type = int64_t;
  // using difference_type = ptrdiff_t;
  using reference = value_type &;
  using const_reference = value_type const &;

 private:
  using pointer = T *;
  pointer data_;
  size_type len_;

 public:
  HOST_DEVICE_INLINE ArrayView(void *data, size_type len)
      : data_(static_cast<pointer>(data)), len_(len) {}

  HOST_DEVICE_INLINE const_reference operator[](size_type index) const { return data_[index]; }

  HOST_DEVICE_INLINE reference operator[](size_type index) { return data_[index]; }

  HOST_DEVICE_INLINE size_type &size() { return len_; }
};
}  // namespace embedding

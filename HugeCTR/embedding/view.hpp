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
struct DefaultPtrTraits {
  using pointer = T *;
};

template <typename T>
struct RestrictPtrTraits {
  using pointer = T *__restrict__;
};

template <typename T, template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class ArrayView {
 public:
  using value_type = T;
  using size_type = index_t;
  // using difference_type = ptrdiff_t;
  using reference = value_type &;
  using const_reference = value_type const &;

 private:
  using pointer = typename PtrTraits<T>::pointer;
  pointer data_;
  size_type len_;

 public:
  HOST_DEVICE_INLINE ArrayView(void *data, size_type len)
      : data_(static_cast<pointer>(data)), len_(len) {}

  HOST_DEVICE_INLINE const_reference operator[](size_type index) const { return data_[index]; }

  HOST_DEVICE_INLINE reference operator[](size_type index) { return data_[index]; }

  HOST_DEVICE_INLINE size_type &size() { return len_; }
};

template <typename T, typename index_t, typename Lambda>
class LambdaIterator {
 public:
  using value_type = T;
  using size_type = index_t;

 private:
  Lambda lambda_;
  size_type len_;

 public:
  HOST_DEVICE_INLINE LambdaIterator(Lambda lambda, size_type len) : lambda_(lambda), len_(len) {}

  HOST_DEVICE_INLINE value_type operator[](size_type index) const { return lambda_(index); }

  HOST_DEVICE_INLINE size_type &size() { return len_; }
};

template <typename emb_t, template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class RaggedEmbForwardResultView {
 public:
  using value_type = ArrayView<emb_t, PtrTraits, index_t>;
  using size_type = index_t;

 private:
  using pointer = typename PtrTraits<emb_t>::pointer;
  pointer data_;
  const int *ev_offset_list_;
  int batch_size_per_gpu_;

 public:
  HOST_DEVICE_INLINE RaggedEmbForwardResultView(void *data, const int *ev_offset_list,
                                                int batch_size_per_gpu)
      : data_(static_cast<pointer>(data)),
        ev_offset_list_(ev_offset_list),
        batch_size_per_gpu_(batch_size_per_gpu) {}

  DEVICE_INLINE value_type operator[](size_type idx) {
    int embedding_id = idx / batch_size_per_gpu_;
    int batch_id = idx % batch_size_per_gpu_;

    int ev_size = ev_offset_list_[embedding_id + 1] - ev_offset_list_[embedding_id];
    int start = ev_offset_list_[embedding_id] * batch_size_per_gpu_ + ev_size * batch_id;
    return ArrayView<emb_t, PtrTraits, index_t>{data_ + start, ev_size};
  }
};

}  // namespace embedding

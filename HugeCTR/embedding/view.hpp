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
class RaggedLookupResultView {
 public:
  using value_type = ArrayView<emb_t, PtrTraits, index_t>;
  using size_type = index_t;

 private:
  using pointer = typename PtrTraits<emb_t *>::pointer;
  pointer data_;
  const uint32_t *offset_;
  int num_offset_;
  const int *local_ev_size_list_;
  int batch_size_per_gpu_;

 public:
  HOST_DEVICE_INLINE
  RaggedLookupResultView(void *data, const uint32_t *offset, int num_offset, const int *local_ev_size_list, int batch_size_per_gpu)
      : data_(static_cast<pointer>(data)),
        offset_(offset),
        num_offset_(num_offset),
        local_ev_size_list_(local_ev_size_list),
        batch_size_per_gpu_(batch_size_per_gpu) {}

  DEVICE_INLINE value_type operator[](size_type idx) {
    int bucket_id =
        binary_search_index_lower_bound(offset_, num_offset_, static_cast<uint32_t>(idx));
    int embedding_id = bucket_id / batch_size_per_gpu_;
    int ev_size = local_ev_size_list_[embedding_id];
    return ArrayView<emb_t, PtrTraits, index_t>{data_[idx], ev_size};
  }
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


template <typename emb_t, template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class RaggedModelBufferView {
 public:
  using value_type = ArrayView<emb_t, PtrTraits, index_t>;
  using size_type = index_t;

 private:
  using pointer = typename PtrTraits<emb_t *>::pointer;
  pointer data_;
  const int *local_ev_offset_list_;
  
  int num_gpus_;
  int batch_size_;
  int batch_size_per_gpu_;

 public:
  HOST_DEVICE_INLINE
  RaggedModelBufferView(void *data, const int *local_ev_offset_list, int num_gpus, int batch_size)
      : data_(static_cast<pointer>(data)),
        local_ev_offset_list_(local_ev_offset_list),
        num_gpus_(num_gpus),
        batch_size_(batch_size),
        batch_size_per_gpu_(batch_size / num_gpus) {}

  DEVICE_INLINE value_type operator[](size_type idx) {
    int embedding_id = idx / batch_size_;
    int batch_id = idx % batch_size_;
    int gpu_id = batch_id / batch_size_per_gpu_;
    int local_batch_id = batch_id % batch_size_per_gpu_;
    int ev_size =
        local_ev_offset_list_[embedding_id + 1] - local_ev_offset_list_[embedding_id];
    int start =
        batch_size_per_gpu_ * local_ev_offset_list_[embedding_id] + local_batch_id * ev_size;

    return ArrayView<emb_t, PtrTraits, index_t>{data_[gpu_id] + start, ev_size};
  }
};

template <typename emb_t, template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class RaggedNetworkBufferView {
 public:
  using value_type = ArrayView<emb_t, PtrTraits, index_t>;
  using size_type = index_t;

 private:
  using pointer = typename PtrTraits<emb_t *>::pointer;
  pointer data_;
  const int *gpu_idx_offset_;
  const int **global_ev_offset_;
  
  int num_gpus_;
  int batch_size_;
  int batch_size_per_gpu_;

 public:
  HOST_DEVICE_INLINE
  RaggedNetworkBufferView(void *data, const int *gpu_idx_offset, const int **global_ev_offset, int num_gpus, int batch_size)
      : data_(static_cast<pointer>(data)),
        gpu_idx_offset_(gpu_idx_offset),
        global_ev_offset_(global_ev_offset),
        num_gpus_(num_gpus),
        batch_size_(batch_size),
        batch_size_per_gpu_(batch_size / num_gpus) {}

  DEVICE_INLINE value_type operator[](size_type idx) {
    int gpu_id = 
        binary_search_index_lower_bound(gpu_idx_offset_, num_gpus_ + 1, static_cast<int>(idx));
    int local_bucket_id = idx - gpu_idx_offset_[gpu_id];
    int local_embedding_id = local_bucket_id / batch_size_per_gpu_;
    int local_batch_id = local_bucket_id % batch_size_per_gpu_;
    int ev_size = global_ev_offset_[gpu_id][local_embedding_id + 1] -
                  global_ev_offset_[gpu_id][local_embedding_id];
    int start = global_ev_offset_[gpu_id][local_embedding_id] * batch_size_per_gpu_ + local_batch_id * ev_size;
    return ArrayView<emb_t, PtrTraits, index_t>{data_[gpu_id] + start, ev_size};
  }
};

template <typename emb_t, template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class RaggedGradBufferView {
 public:
  using value_type = ArrayView<emb_t, PtrTraits, index_t>;
  using size_type = index_t;

 private:
  using pointer = typename PtrTraits<emb_t>::pointer;
  pointer data_;
  const uint32_t *ev_size_scan_list_;

 public:
  HOST_DEVICE_INLINE RaggedGradBufferView(void *data, const uint32_t *ev_size_scan_list)
      : data_(static_cast<pointer>(data)),
        ev_size_scan_list_(ev_size_scan_list) {}

  DEVICE_INLINE value_type operator[](size_type idx) {
    int ev_size = ev_size_scan_list_[idx + 1] - ev_size_scan_list_[idx];
    int start = ev_size_scan_list_[idx];
    return ArrayView<emb_t, PtrTraits, index_t>{data_ + start, ev_size};
  }
};
}  // namespace embedding

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
#pragma once

#include <unistd.h>

#include <atomic>
#include <cassert>

namespace HugeCTR {

template <typename T>
class mpmc_bounded_queue {
 public:
  mpmc_bounded_queue(size_t buffer_size)
      : buffer_(new cell_t[buffer_size]),
        buffer_mask_(buffer_size - 1)

  {
    assert((buffer_size >= 2) && ((buffer_size & (buffer_size - 1)) == 0));

    for (size_t i = 0; i != buffer_size; i += 1)
      buffer_[i].sequence_.store(i, std::memory_order_relaxed);

    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
  }

  ~mpmc_bounded_queue() { delete[] buffer_; }

  bool enqueue(T const& data) {
    cell_t* cell;
    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);

    for (;;) {
      cell = &buffer_[pos & buffer_mask_];
      size_t seq = cell->sequence_.load(std::memory_order_acquire);

      intptr_t dif = (intptr_t)seq - (intptr_t)pos;

      if (dif == 0) {
        if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
      } else if (dif < 0)
        return false;
      else
        pos = enqueue_pos_.load(std::memory_order_relaxed);
    }

    cell->data_ = data;
    cell->sequence_.store(pos + 1, std::memory_order_release);

    return true;
  }

  bool dequeue(T& data) {
    cell_t* cell;
    size_t pos = dequeue_pos_.load(std::memory_order_relaxed);

    for (;;) {
      cell = &buffer_[pos & buffer_mask_];
      size_t seq = cell->sequence_.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);

      if (dif == 0) {
        if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
      } else if (dif < 0)
        return false;
      else
        pos = dequeue_pos_.load(std::memory_order_relaxed);
    }

    data = cell->data_;

    cell->sequence_.store(pos + buffer_mask_ + 1, std::memory_order_release);

    return true;
  }

 private:
  struct cell_t {
    std::atomic<size_t> sequence_;
    T data_;
  };

  static size_t const cacheline_size = 64;
  typedef char cacheline_pad_t[cacheline_size];
  cacheline_pad_t pad0_;
  cell_t* const buffer_;
  size_t const buffer_mask_;
  cacheline_pad_t pad1_;
  std::atomic<size_t> enqueue_pos_;
  cacheline_pad_t pad2_;
  std::atomic<size_t> dequeue_pos_;
  cacheline_pad_t pad3_;
};

template <typename T>
class WorkQueue {
 public:
  WorkQueue(size_t queue_depth)
      : queue_(
            std::max(2ul, (size_t)pow(2, ceil(log(queue_depth) /
                                              log(2)))))  // round queue depth up to next power of 2
  {}

  void push(const T& elem) {
    while (!queue_.enqueue(elem)) {
      usleep(10);
    }
  }

  bool try_pop(T& elem) { return queue_.dequeue(elem); }

 private:
  mpmc_bounded_queue<T> queue_;
};

}  // namespace HugeCTR
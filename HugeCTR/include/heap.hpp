/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <thread>
#include "HugeCTR/include/common.hpp"

namespace HugeCTR {

/**
 * @brief A thread safe heap implementation for multiple reading and writing.
 *
 * An requirment of HugeCTR is multiple data reading threads for
 * the sake of high input throughput. The specific chunk in a heap
 * will be locked while it's in use by one of the threads, and will be
 * unlocked when it's checkin.
 * Note that at most 32 chunks are avaliable for a heap.
 */
template <typename T>
class Heap {
 private:
  // set to statue 0, while initilization convert some of the lower bits to 1;
  unsigned int higher_bits_{0}, lower_bits_{0};
  std::vector<T> chunks_;
  std::mutex mtx_;
  std::condition_variable write_cv_, read_cv_;
  std::atomic<bool> loop_flag_{true};

 public:
  /**
   * Writer's free chunk checkout.
   * Get a freed or idle chunk from heap. Users can write data into it then.
   * Heap will block checkout while no avaliable chunk.
   * @param chunk the pointer of the chunk passed out.
   * @param key the id of the chunk, which will be used when checkin.
   */
  void free_chunk_checkout(T** chunk, unsigned int* key) {
    if (chunk == nullptr || key == nullptr) {
      CK_THROW_(Error_t::WrongInput, "chunk == nullptr || key == nullptr");
    }
    std::unique_lock<std::mutex> lock(mtx_);
    // thread safe start
    while (loop_flag_) {
      int id = __builtin_ffs(lower_bits_ & (~higher_bits_)) - 1;
      if (id >= 0) {
        lower_bits_ &= (~(1u << id));
        *chunk = &chunks_[id];
        *key = 1u << id;
        break;
      }
      read_cv_.wait(lock);
    }
    //if (!loop_flag_){ std::cout << "loop_flag2_" << std::endl;}
    // thread safe end
    return;
  }

  /**
   * Writer's chunk checkin.
   * After user write data into this chunk. User (writer) should call this function to
   * Checkin the chunk so that the reader can get the prepared chunk and read.
   * @param key the id of the chunk, returned by free_chunk_checkout().
   */
  void chunk_write_and_checkin(unsigned int key) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      // thread safe start
      lower_bits_ |= key;
      higher_bits_ |= key;
      // thread safe end
    }
    write_cv_.notify_one();
    return;
  }

  /**
   * Readers's chunk checkout.
   * Readers checkout a ready chunk, which is checkin by writer.
   * If no ready chunk availiable, it will do spin lock.
   * @param chunk chunk of data ready to read.
   * @param key the id of the chunk, which will be used when checkin.
   */
  void data_chunk_checkout(T** chunk, unsigned int* key) {
    if (chunk == nullptr || key == nullptr) {
      CK_THROW_(Error_t::WrongInput, "chunk == nullptr || key == nullptr");
    }

    std::unique_lock<std::mutex> lock(mtx_);
    // thread safe start
    while (loop_flag_) {
      int id = __builtin_ffs(lower_bits_ & higher_bits_) - 1;
      if (id >= 0) {
        lower_bits_ &= (~(1u << id));
        *chunk = &chunks_[id];
        *key = 1u << id;
        break;
      }
      write_cv_.wait(lock);
    }
    //if (!loop_flag_){ std::cout << "loop_flag1_" << std::endl;}
    // thread safe end
    return;
  }

  /**
   * Reader's free chunk and checkin.
   * After consuming the data and ensure no use in the future, reader can
   * call this methord to free the chunk and checkin to tell the writer that
   * the chunk can be used by writer again.
   * @param key the id of the chunk.
   */
  void chunk_free_and_checkin(unsigned int key) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      // thread safe start
      lower_bits_ |= key;
      higher_bits_ &= (~key);
      // thread safe end
    }
    read_cv_.notify_one();
    return;
  }

  /**
   * break the spin lock.
   */
  void break_and_return() {
    loop_flag_ = false;
    write_cv_.notify_all();
    read_cv_.notify_all();
    return;
  }

  /**
   * Ctor.
   * Make "num" copy of the chunks.
   */
  template <typename... Args>
  Heap(int num, Args&&... args) {
    if (num > static_cast<int>(sizeof(unsigned int) * 8)) {
      CK_THROW_(Error_t::OutOfBound, "num > sizeof(unsigned int) * 8");
    } else if (num <= 0) {
      CK_THROW_(Error_t::WrongInput, "num <= 0");
    }
    for (int i = 0; i < num; i++) {
      chunks_.emplace_back(T(std::forward<Args>(args)...));
    }
    lower_bits_ = (1ull << num) - 1;
  }
};

}  // namespace HugeCTR

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
class HeapEx {
 private:
  unsigned int higher_bits_{0}, lower_bits_{0};
  const int num_chunks_;
  std::vector<T> chunks_;
  std::mutex mtx_;
  std::condition_variable write_cv_, read_cv_;
  std::atomic<bool> loop_flag_{true};
  int count_{0};
 public:
  

  /**
   * will try to checkout the chunk id%#chunck
   * if not avaliable just hold
   */
  void free_chunk_checkout(T** chunk, unsigned int id) {
    if (chunk == nullptr) {
      CK_THROW_(Error_t::WrongInput, "chunk == nullptr");
    }
    //    std::thread::id tid = std::this_thread::get_id();
    std::unique_lock<std::mutex> lock(mtx_);
    int i = id%num_chunks_;
    // thread safe start
    //    std::cout << "free_chunk_checkout start" << tid << std::endl;
    while (loop_flag_) {
      bool avail = ((lower_bits_ & (~higher_bits_)) >> i) & 1;
      if (avail) {
        lower_bits_ &= (~(1u << i));
        *chunk = &chunks_[i];
        break;
      }
      read_cv_.wait(lock);
    }
    //    std::cout << "free_chunk_checkout end " << loop_flag_ << " " << avail << " " << tid << std::endl;
    return;
  }
  
  /**
   * After writting, check in the chunk
   */
  void chunk_write_and_checkin(unsigned int id) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      int i = id%num_chunks_;
      // thread safe start
      lower_bits_ |= (1u << i);
      higher_bits_ |= (1u << i);
      // thread safe end
    }
    write_cv_.notify_all();
    // std::thread::id tid = std::this_thread::get_id();
    // std::cout << "chunk_write_and_checkin" << tid << std::endl;
    return;
  }

  /**
   * Checkout the data with id count_%chunk
   * if not avaliable hold
   */
  void data_chunk_checkout(T** chunk) {
    if (chunk == nullptr) {
      CK_THROW_(Error_t::WrongInput, "chunk == nullptr");
    }
    // std::thread::id tid = std::this_thread::get_id();
    // std::cout << "data_chunk_checkout start" << tid << std::endl;
    std::unique_lock<std::mutex> lock(mtx_);
    int i = count_;
    // thread safe start
    while (loop_flag_) {
      bool avail = ((lower_bits_ & higher_bits_) >> i) & 1;
      if (avail) {
        lower_bits_ &= (~(1u << i));
        *chunk = &chunks_[i];
        break;
      }
      write_cv_.wait(lock);
    }
    //    std::cout << "data_chunk_checkout end" << tid << std::endl;
    return;
  }

  /**
   * Free the chunk count_%chunk after using.
   */
  void chunk_free_and_checkin() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      // thread safe start
      lower_bits_ |= (1u << count_);
      higher_bits_ &= (~(1u << count_));
      // thread safe end
    }
    read_cv_.notify_all();
    count_ = (count_+1)%num_chunks_;
    // std::thread::id tid = std::this_thread::get_id();
    // std::cout << "chunk_free_and_checkin" << tid << std::endl;
    return;
  }

  /**
   * break the spin lock.
   */
  void break_and_return() {
    //    std::cout << "call break_and_return()" << std::endl;
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
  HeapEx(int num, Args&&... args):num_chunks_(num) {
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

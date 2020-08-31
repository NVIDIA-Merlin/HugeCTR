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
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <common.hpp>

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
  const int num_threads_;

  std::vector<T*> chunks_;
  std::vector<std::queue<T*>> ready_queue_;
  std::vector<std::queue<T*>> wait_queue_;
  std::vector<std::queue<T*>> credits_;
  std::vector<std::condition_variable> read_cond_;
  std::vector<std::condition_variable> write_cond_;

  std::vector<std::mutex> mtx_;
  int count_{0};

 public:
  /**
   * will try to checkout the chunk id%#chunck
   * if not avaliable just hold
   */
  void free_chunk_checkout(T** chunk, unsigned int id) {
    std::unique_lock<std::mutex> lock(mtx_[id]);
    id = id % num_threads_;
    read_cond_[id].wait(lock, [this, &id]() { return !credits_[id].empty(); });
    auto& cand = credits_[id].front();
    wait_queue_[id].push(cand);
    *chunk = cand;
    credits_[id].pop();
    lock.unlock();
  }

  /**
   * After writting, check in the chunk
   */
  void chunk_write_and_checkin(unsigned int id) {
    std::unique_lock<std::mutex> lock(mtx_[id]);
    id = id % num_threads_;
    auto& cand = wait_queue_[id].front();
    ready_queue_[id].push(cand);
    wait_queue_[id].pop();
    lock.unlock();
    write_cond_[id].notify_one();
  }

  /**
   * Checkout the data with id count_%chunk
   * if not avaliable hold
   */
  void data_chunk_checkout(T** chunk) {
    int id = count_;
    std::unique_lock<std::mutex> lock(mtx_[id]);
    write_cond_[id].wait(lock, [this, &id]() { return !ready_queue_[id].empty(); });
    auto& cand = ready_queue_[id].front();
    *chunk = cand;
    lock.unlock();
  }

  /**
   * Free the chunk count_%chunk after using.
   */
  void chunk_free_and_checkin() {
    int id = count_;
    std::unique_lock<std::mutex> lock(mtx_[id]);
    credits_[id].push(ready_queue_[id].front());
    ready_queue_[id].pop();
    lock.unlock();
    read_cond_[id].notify_one();
    count_ = (id + 1) % num_threads_;
    return;
  }

  /**
   * break the spin lock.
   */
  void break_and_return() {
    for (int id = 0; id < num_threads_; id++) {
      ready_queue_[id].push(nullptr);
      credits_[id].push(nullptr);
      write_cond_[id].notify_one();
      read_cond_[id].notify_one();
    }
    return;
  }

  /**
   * Ctor.
   * Make "num" copy of the chunks.
   */
  template <typename... Args>
  HeapEx(int num, Args&&... args)
      : num_threads_(num),
        ready_queue_(num),
        wait_queue_(num),
        credits_(num),
        read_cond_(num),
        write_cond_(num),
        mtx_(num) {
    if (num > static_cast<int>(sizeof(unsigned int) * 8)) {
      CK_THROW_(Error_t::OutOfBound, "num > sizeof(unsigned int) * 8");
    } else if (num <= 0) {
      CK_THROW_(Error_t::WrongInput, "num <= 0");
    }

    for (int i = 0; i < num; i++) {
      for (int j = 0; j < 1; j++) {
        chunks_.emplace_back(new T(std::forward<Args>(args)...));
        credits_[i].emplace(chunks_.back());
      }
    }
  }
  
  int get_size(){
    return num_threads_;
  }

  ~HeapEx() {
    for (size_t i = 0; i < chunks_.size(); i++) {
      T* cand = chunks_[i];
      delete cand;
    }
  }
};

}  // namespace HugeCTR

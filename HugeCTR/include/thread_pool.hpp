/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace HugeCTR {

class ThreadPool {
 public:
  ThreadPool();

  ThreadPool(const ThreadPool&) = delete;

  ThreadPool(size_t num_workers);

  virtual ~ThreadPool();

  ThreadPool& operator=(const ThreadPool&) = delete;

  size_t size() const { return workers_.size(); }

  bool idle() const;

  void await_idle() const;

  std::future<void> submit(std::function<void()> task);

  static ThreadPool& get();

  template <typename TInputIterator>
  static void await(TInputIterator first, const TInputIterator& last);

 private:
  std::vector<std::thread> workers_;

  mutable std::mutex barrier_;  // Must be obtained to ensure exclusive access.
  mutable std::condition_variable
      submit_sempahore_;  // Triggered on submission. Workers wait for this.
  mutable std::condition_variable idle_semaphore_;  // Trigger

  bool terminate_ = false;  // Used to signal to the workers that termination is imminent.
  size_t num_idle_workers_ = 0;
  std::deque<std::packaged_task<void()>>
      packages_;  // Work packages that have not been processed yet.

  void run_(const size_t thread_index);
};

template <typename TInputIterator>
void ThreadPool::await(TInputIterator first, const TInputIterator& last) {
  for (; first != last; first++) {
    first->wait();
  }
}

}  // namespace HugeCTR
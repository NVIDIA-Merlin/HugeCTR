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

#include <cstdlib>
#include <iostream>
#include <thread_pool.hpp>

namespace HugeCTR {

static std::unique_ptr<ThreadPool> DEFAULT_THREAD_POOL;

ThreadPool::ThreadPool() : ThreadPool(0) {}

ThreadPool::ThreadPool(size_t num_threads) : terminate_(false) {
  // Determine eventual number of threads.
  if (num_threads == 0) {
    const char* num_threads_str = getenv("HCTR_DEFAULT_CONCURRENCY");
    if (num_threads_str) {
      num_threads = std::stoull(num_threads_str);
    } else {
      num_threads = std::thread::hardware_concurrency();
    }
  }

  // Create threads.
  for (size_t thread_num = 0; thread_num < num_threads; thread_num++) {
    pool_.emplace_back(&ThreadPool::run, this, thread_num);
  }
}

ThreadPool::~ThreadPool() {
  terminate_ = true;
  sempahore_.notify_all();
  for (auto& thread : pool_) {
    thread.join();
  }
}

size_t ThreadPool::size() const { return pool_.size(); }

ThreadPoolResult ThreadPool::post(ThreadPoolTask task) {
  std::packaged_task<void(size_t, size_t)> actual_task(std::move(task));
  ThreadPoolResult result = actual_task.get_future();
  {
    std::unique_lock<std::mutex> lock(queue_guard_);
    queue_.push_back(std::move(actual_task));
  }
  sempahore_.notify_one();
  return result;
}

void ThreadPool::await(std::vector<ThreadPoolResult>& results) {
  for (const auto& result : results) {
    result.wait();
  }
}

ThreadPool& ThreadPool::get() {
  if (!DEFAULT_THREAD_POOL) {
    DEFAULT_THREAD_POOL = std::make_unique<ThreadPool>();
  }
  return *DEFAULT_THREAD_POOL.get();
}

void ThreadPool::run(const size_t thread_num) {
  const size_t num_threads = pool_.size();
  while (!terminate_) {
    thread_local std::packaged_task<void(size_t, size_t)> task;
    {
      std::unique_lock<std::mutex> lock(queue_guard_);
      sempahore_.wait(lock, [&] { return terminate_ || queue_.size(); });
      if (terminate_) {
        break;
      }
      task = std::move(queue_.front());
      queue_.pop_front();
    }
    task(thread_num, num_threads);
  }
}

}  // namespace HugeCTR
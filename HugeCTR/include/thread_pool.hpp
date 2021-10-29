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
#include <mutex>
#include <thread>
#include <vector>

namespace HugeCTR {

using ThreadPoolTask = std::function<void(size_t, size_t)>;
using ThreadPoolResult = std::future<void>;

class ThreadPool {
 public:
  ThreadPool();

  ThreadPool(const ThreadPool&) = delete;

  ThreadPool(size_t num_threads);

  virtual ~ThreadPool();

  ThreadPool& operator=(const ThreadPool&) = delete;

  size_t size() const;

  ThreadPoolResult post(ThreadPoolTask task);

  static void await(std::vector<ThreadPoolResult>& results);

  static ThreadPool& get();

 private:
  bool terminate_;
  std::vector<std::thread> pool_;
  std::condition_variable sempahore_;
  std::mutex queue_guard_;
  std::deque<std::packaged_task<void(size_t, size_t)>> queue_;

  void run(const size_t thread_num);
};

}  // namespace HugeCTR
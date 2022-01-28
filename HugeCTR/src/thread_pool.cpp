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

#include <base/debug/logger.hpp>
#include <cstdlib>
#include <iostream>
#include <thread_pool.hpp>

namespace HugeCTR {

ThreadPool::ThreadPool(const std::string& name) : ThreadPool(name, 0) {}

ThreadPool::ThreadPool(const std::string& name, size_t num_workers) : name_(name) {
  // Determine eventual number of threads.
  if (num_workers == 0) {
    const char* num_workers_str = getenv("HCTR_DEFAULT_CONCURRENCY");
    if (num_workers_str) {
      num_workers = std::stoull(num_workers_str);
    } else {
      num_workers = std::thread::hardware_concurrency();
    }
  }

  // Create worker threads.
  for (size_t i = 0; i < num_workers; i++) {
    workers_.emplace_back(&ThreadPool::run_, this, i);
  }

  // Block until all workers entered the idle state.
  await_idle();
}

ThreadPool::~ThreadPool() {
  // Momentarily request exclusive access, and set terminate condition.
  {
    std::lock_guard<std::mutex> lock(barrier_);
    terminate_ = true;
    submit_sempahore_.notify_all();
  }

  // Wait for the worker threads to exit.
  for (auto& worker : workers_) {
    worker.join();
  }
}

bool ThreadPool::idle() const {
  // Momentarily request exclusive access, and read out the idle status.
  std::lock_guard<std::mutex> lock(barrier_);
  return num_idle_workers_ == workers_.size() && packages_.empty();
}

void ThreadPool::await_idle() const {
  // Momentarily request exclusive access.
  std::unique_lock<std::mutex> lock(barrier_);

  // Are we idle already? If not wait for a worker to exit.
  while (num_idle_workers_ != workers_.size() || !packages_.empty()) {
    HCTR_THROW_IF(terminate_, Error_t::IllegalCall,
                  "Attempted to await an already terminated ThreadPool!");
    idle_semaphore_.wait(lock);
  }
}

std::future<void> ThreadPool::submit(std::function<void()> task) {
  std::packaged_task<void()> package(std::move(task));
  std::future<void> result = package.get_future();

  // Momentarily request exclusive access, to submit the task.
  {
    std::lock_guard<std::mutex> lock(barrier_);
    HCTR_THROW_IF(terminate_, Error_t::IllegalCall,
                  "Attempted to submit work to an already terminated ThreadPool!");
    packages_.push_back(std::move(package));
  }

  // Wake up a worker.
  submit_sempahore_.notify_one();

  return result;
}

ThreadPool& ThreadPool::get() {
  // Lazy init of default thread-pool on first call to this function..
  static std::unique_ptr<ThreadPool> default_pool;
  static std::once_flag semaphore;
  call_once(semaphore, []() { default_pool = std::make_unique<ThreadPool>("default"); });
  return *default_pool.get();
}

void ThreadPool::run_(const size_t thread_index) {
  hctr_set_thread_name(name_ + " #" + std::to_string(thread_index));
  while (true) {
    thread_local std::packaged_task<void()> package;

    // Acquire exclusive access.
    {
      std::unique_lock<std::mutex> lock(barrier_);

      // If termination request occured.
      if (terminate_) {
        return;
      }

      // If no work package queued.
      while (packages_.empty()) {
        // Enter idle state (notify threads that wait for the threadpool to go idle).
        num_idle_workers_ += 1;
        idle_semaphore_.notify_all();

        // Wait for a task.
        submit_sempahore_.wait(lock);
        num_idle_workers_ -= 1;

        // If woken up by terminate request.
        if (terminate_) {
          return;
        }
      }
      package = std::move(packages_.front());
      packages_.pop_front();
    }

    // Execute work package.
    package();
  }
}

}  // namespace HugeCTR
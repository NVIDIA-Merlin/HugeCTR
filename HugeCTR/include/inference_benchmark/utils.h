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

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace HugeCTR {

enum ProfilerType_t { Timeliness, Occupancy, Event };
enum ProfilerTarget_t { EC, HPSBACKEND, LOOKSESSION, INFERENCESESSION, REST };

struct metrics_argues {
  metrics_argues()
      : embedding_cache(false), database_backend(false), lookup_session(false), refresh_ec(false) {}
  int table_size;
  int iterations;
  int warmup;
  bool embedding_cache;
  bool database_backend;
  bool lookup_session;
  bool refresh_ec;
  int num_keys;
};

struct Profiler_Config {
  Profiler_Config() : enable_bench_(false) {}
  int interations;
  int warmup;
  bool enable_bench_;
};

class BaseUnit {
 public:
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual double getValue() const = 0;
};

class CPUTimer : public BaseUnit {
 public:
  CPUTimer() {}
  void start() override {
    m_StartTime_ = std::chrono::high_resolution_clock::now();
    m_bRunning_ = true;
  }
  void stop() override {
    m_EndTime_ = std::chrono::high_resolution_clock::now();
    m_bRunning_ = false;
  }
  double getValue() const override {
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime;
    if (m_bRunning_) {
      endTime = std::chrono::high_resolution_clock::now();
    } else {
      endTime = m_EndTime_;
    }
    return std::chrono::duration<double, std::milli>(endTime - m_StartTime_).count();
  }

  ~CPUTimer() {}

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTime_{};
  std::chrono::time_point<std::chrono::high_resolution_clock> m_EndTime_{};
  bool m_bRunning_ = false;
};

class Scalar : public BaseUnit {
 public:
  Scalar() {}
  Scalar(double value) { value_ = value; }
  void start(double value) { value_ = value; }
  void start() override {}
  void stop() override {}
  double getValue() const override { return value_; }

  ~Scalar() {}

 private:
  double value_;
};

class Lvector {
  std::shared_mutex m_mutex;
  std::vector<BaseUnit *> v_vector;

 public:
  Lvector() {}
  Lvector(std::vector<BaseUnit *> ref) { v_vector.assign(ref.begin(), ref.end()); }
  BaseUnit *operator[](int idx) { return v_vector[idx]; }
  BaseUnit *get(int idx) { return v_vector[idx]; }
  void push_back(BaseUnit *elem) {
    std::lock_guard lock(m_mutex);
    v_vector.push_back(elem);
  }
  size_t get_size() { return v_vector.size(); }
};

class BaseIndicator {
  using mutex_type = std::shared_timed_mutex;
  using read_only_lock = std::shared_lock<mutex_type>;
  using updatable_lock = std::unique_lock<mutex_type>;

 private:
  mutex_type mtx_for_m;
  std::map<std::string, std::shared_ptr<Lvector>> indicator_map_;
  ProfilerType_t indicator_type_;
  size_t warmup_iters;

 public:
  BaseIndicator() {}
  BaseIndicator(ProfilerType_t indicator_type, int warmup_iters = 0) {
    indicator_type_ = indicator_type;
    warmup_iters = warmup_iters;
  }

  void set_warminter(int warmup_iter) { warmup_iters = warmup_iter; }

  void put(std::string &key, Lvector *vector) {
    updatable_lock lock(mtx_for_m);
    indicator_map_[key] = std::shared_ptr<Lvector>(vector);
  }

  std::shared_ptr<Lvector> get(std::string &key) {
    read_only_lock lock(mtx_for_m);
    auto it = indicator_map_.find(key);
    if (it != indicator_map_.end()) {
      return it->second;
    }
    return nullptr;
  }

  bool remove(std::string &key) {
    updatable_lock lock(mtx_for_m);
    auto n = indicator_map_.erase(key);
    return n;
  }
  void print() {
    if (indicator_type_ == ProfilerType_t::Timeliness) {
      for (auto it : indicator_map_) {
        size_t v_size = it.second->get_size();
        std::vector<float> results;
        results.reserve(v_size);
        std::cout << "The Benchmark of: " << it.first << std::endl;
        for (size_t i = warmup_iters > v_size ? 0 : warmup_iters; i < v_size; i++) {
          results.push_back(it.second->get(i)->getValue());
        }
        std::sort(results.begin(), results.end());
        std::cout << "Latencies "
                  << "[" << results.size() << " iterations] "
                  << "min = " << *(results.begin()) << "ms, "
                  << "mean = "
                  << std::accumulate(results.begin(), results.end(), 0.f) / results.size() << "ms, "
                  << "median = " << results[results.size() / 2] << "ms, "
                  << "95% = " << results[size_t(0.95 * results.size())] << "ms, "
                  << "99% = " << results[size_t(0.99 * results.size())] << "ms, "
                  << "max = " << *(results.rbegin()) << "ms, "
                  << "throughput = " << 1000 / (results[results.size() / 2]) << "/s" << std::endl;
      }
    }
    if (indicator_type_ == ProfilerType_t::Occupancy) {
      for (auto it : indicator_map_) {
        size_t v_size = it.second->get_size();
        std::vector<float> results;
        results.reserve(v_size);
        std::cout << "The Benchmark of: " << it.first << std::endl;
        for (size_t i = warmup_iters > v_size ? 0 : warmup_iters; i < v_size; i++) {
          results.push_back(it.second->get(i)->getValue());
        }
        std::sort(results.begin(), results.end());
        std::cout << "Occupancy "
                  << "[" << results.size() << " iterations] "
                  << "min = " << *(results.begin()) << ", "
                  << "mean = "
                  << std::accumulate(results.begin(), results.end(), 0.f) / results.size() << ", "
                  << "median = " << results[results.size() / 2] << ", "
                  << "95% = " << results[size_t(0.95 * results.size())] << ", "
                  << "99% = " << results[size_t(0.99 * results.size())] << ", "
                  << "max = " << *(results.rbegin()) << std::endl;
      }
    }
  }
};

}  // namespace HugeCTR
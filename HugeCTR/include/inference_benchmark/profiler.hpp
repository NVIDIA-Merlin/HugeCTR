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

#include <inference_benchmark/utils.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace HugeCTR {

/// Holds the server-side inference statisitcs of the target model and its
/// composing models
#ifdef BENCHMARK_HPS
class profiler {
 public:
  profiler(ProfilerTarget_t traget = ProfilerTarget_t::REST) {
    profiler_target_ = traget;
    Timeliness = std::make_unique<BaseIndicator>(ProfilerType_t::Timeliness);
    Occupancy_models_stat = std::make_unique<BaseIndicator>(ProfilerType_t::Occupancy);
  };
  ~profiler(){};

  static BaseUnit* start(double scalar = 0, ProfilerType_t type = ProfilerType_t::Timeliness) {
    if (type == ProfilerType_t::Timeliness) {
      CPUTimer* now = new CPUTimer();
      now->start();
      return now;
    }
    if (type == ProfilerType_t::Occupancy) {
      Scalar* value = new Scalar(scalar);
      return value;
    }
    return nullptr;
  };

  void end(BaseUnit* start, std::string indicator_name,
           ProfilerType_t type = ProfilerType_t::Timeliness, cudaStream_t st = 0) {
    if (st != 0) {
      cudaStreamSynchronize(st);
    }
    if (config_.enable_bench_ && type == ProfilerType_t::Timeliness) {
      start->stop();
      if (Timeliness->get(indicator_name) == nullptr) {
        Lvector* temp = new Lvector();
        temp->push_back(start);
        Timeliness->put(indicator_name, temp);
      } else {
        Timeliness->get(indicator_name)->push_back(start);
      }
    }
    if (config_.enable_bench_ && type == ProfilerType_t::Occupancy) {
      if (Occupancy_models_stat->get(indicator_name) == nullptr) {
        Lvector* temp = new Lvector();
        temp->push_back(start);
        Occupancy_models_stat->put(indicator_name, temp);
      } else {
        Occupancy_models_stat->get(indicator_name)->push_back(start);
      }
    }
  };

  void set_config(int interation, int warmup, bool enable_bench) {
    config_.interations = interation;
    config_.warmup = warmup;
    config_.enable_bench_ = enable_bench;
    Timeliness->set_warminter(warmup);
    Occupancy_models_stat->set_warminter(warmup);
  }

  void print() {
    if (config_.enable_bench_) {
      Timeliness->print();
      Occupancy_models_stat->print();
    }
  }

 private:
  std::unique_ptr<BaseIndicator> Occupancy_models_stat;
  // std::unique_ptr<BaseIndicator> Event_models_stat;
  std::unique_ptr<BaseIndicator> Timeliness;
  ProfilerTarget_t profiler_target_;
  bool enable_bench_ = false;
  Profiler_Config config_;
};

#else
class profiler {
 public:
  profiler(ProfilerTarget_t traget = ProfilerTarget_t::REST) {
    profiler_target_ = traget;
    Timeliness = std::make_unique<BaseIndicator>(ProfilerType_t::Timeliness);
    Occupancy_models_stat = std::make_unique<BaseIndicator>(ProfilerType_t::Occupancy);
  };
  ~profiler(){};

  static BaseUnit* start(double scalar = 0, ProfilerType_t type = ProfilerType_t::Timeliness) {
    return nullptr;
  };

  void end(BaseUnit* start, std::string indicator_name,
           ProfilerType_t type = ProfilerType_t::Timeliness, cudaStream_t st = 0){};

  void set_config(int interation, int warmup, bool enable_bench) {
    config_.enable_bench_ = enable_bench;
    enable_bench_ = enable_bench;
  };

  void print() {
    std::cout << "Please add ENABLE_PROFILER build option for HPS profiler." << std::endl;
  }

 private:
  std::unique_ptr<BaseIndicator> Occupancy_models_stat;
  // std::unique_ptr<BaseIndicator> Event_models_stat;
  std::unique_ptr<BaseIndicator> Timeliness;
  ProfilerTarget_t profiler_target_;
  bool enable_bench_ = false;
  Profiler_Config config_;
};
#endif

}  // namespace HugeCTR

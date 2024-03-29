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

#include <common.hpp>
#include <functional>
#include <gpu_resource.hpp>
#include <graph_wrapper.hpp>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace HugeCTR {

class Scheduleable {
 public:
  virtual ~Scheduleable() = default;

  virtual void init(std::shared_ptr<GPUResource> gpu){};

  virtual void run(std::shared_ptr<GPUResource> gpu, bool use_graph) = 0;
};

class StreamContextScheduleable : public Scheduleable {
 private:
  std::optional<std::string> stream_name_;
  int priority_;
  bool is_absolute_stream_;
  std::optional<std::vector<cudaEvent_t>> schedule_event_;
  bool wait_external_;
  std::optional<cudaEvent_t> completion_event_;
  bool record_external_;

  std::function<void()> workload_;

 public:
  HCTR_DISALLOW_COPY_AND_MOVE(StreamContextScheduleable);

  explicit StreamContextScheduleable(std::function<void()> workload);

  ~StreamContextScheduleable() override;

  void set_absolute_stream(const std::string &stream_name, int priority = 0);

  void set_stream(const std::string &stream_name, int priority = 0);

  std::tuple<std::string, int> get_stream_name(std::shared_ptr<GPUResource> gpu);

  void wait_event(const std::vector<cudaEvent_t> &schedule_event, bool external = false);

  cudaEvent_t record_done(bool external = false, unsigned int flags = cudaEventDisableTiming);

  void init(std::shared_ptr<GPUResource> gpu) override;

  void run(std::shared_ptr<GPUResource> gpu, bool use_graph) override;
};

class GraphScheduleable : public Scheduleable {
 private:
  std::vector<std::shared_ptr<Scheduleable>> scheduleable_list_;
  GraphWrapper graph_;

 public:
  HCTR_DISALLOW_COPY_AND_MOVE(GraphScheduleable);

  template <typename... T>
  GraphScheduleable(std::shared_ptr<T>... scheduleable) {
    static_assert(std::conjunction<std::is_base_of<Scheduleable, T>...>::value, "");
    (scheduleable_list_.push_back(std::dynamic_pointer_cast<Scheduleable>(scheduleable)), ...);
  }

  GraphScheduleable(std::vector<std::shared_ptr<Scheduleable>> scheduleable_list)
      : scheduleable_list_(scheduleable_list) {}

  void run(std::shared_ptr<GPUResource> gpu, bool use_graph) override;
};

class Pipeline {
 private:
  std::string stream_name_;
  std::shared_ptr<GPUResource> gpu_resource_;
  std::vector<std::shared_ptr<Scheduleable>> scheduleable_list_;

 public:
  Pipeline() = default;

  Pipeline(const std::string &stream_name, std::shared_ptr<GPUResource> gpu_resource,
           const std::vector<std::shared_ptr<Scheduleable>> &scheduleable_list);

  std::string get_stream_name() { return stream_name_; }

  void run();

  void run_graph();
};
}  // namespace HugeCTR
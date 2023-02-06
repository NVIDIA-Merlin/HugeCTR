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
#include <gpu_resource.hpp>
#include <graph_wrapper.hpp>
#include <string>

namespace HugeCTR {

class Scheduleable {
 protected:
  std::optional<std::string> stream_name_;
  int priority_;
  bool is_absolute_stream_;
  std::optional<std::vector<cudaEvent_t>> schedule_event_;
  bool wait_external_;
  std::optional<cudaEvent_t> completion_event_;
  bool record_external_;

 public:
  explicit Scheduleable();

  void set_absolute_stream(const std::string &stream_name, int priority = 0);

  void set_stream(const std::string &stream_name, int priority = 0);

  void wait_event(const std::vector<cudaEvent_t> &schedule_event, bool external = false);

  cudaEvent_t record_done(bool external = false, unsigned int flags = cudaEventDisableTiming);

  virtual void run(std::shared_ptr<GPUResource> gpu, bool use_graph, bool prepare_resource) = 0;
};

class StreamContextScheduleable : public Scheduleable {
 private:
  std::function<void()> workload_;

 public:
  explicit StreamContextScheduleable(std::function<void()> workload);

  void run(std::shared_ptr<GPUResource> gpu, bool use_graph, bool prepare_resource) override;
};

class Pipeline {
 private:
  std::string stream_name_;
  std::shared_ptr<GPUResource> gpu_resource_;
  cudaStream_t stream_;
  std::vector<std::shared_ptr<Scheduleable>> scheduleable_list_;

  GraphWrapper graph_;

 public:
  Pipeline() = default;

  Pipeline(const std::string &stream_name, std::shared_ptr<GPUResource> gpu_resource,
           const std::vector<std::shared_ptr<Scheduleable>> &scheduleable_list);

  std::string get_stream_name() { return stream_name_; }

  void run();

  void run_graph();
};

}  // namespace HugeCTR
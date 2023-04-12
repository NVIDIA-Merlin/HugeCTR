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

#include <unistd.h>

#include <pipeline.hpp>

namespace HugeCTR {

StreamContextScheduleable::StreamContextScheduleable(std::function<void()> workload)
    : stream_name_(std::nullopt),
      priority_(0),
      is_absolute_stream_(false),
      schedule_event_(std::nullopt),
      wait_external_(false),
      completion_event_(std::nullopt),
      record_external_(false),
      workload_(workload) {}

StreamContextScheduleable::~StreamContextScheduleable() {
  if (completion_event_) {
    cudaEventDestroy(completion_event_.value());
  }
}

void StreamContextScheduleable::set_absolute_stream(const std::string &stream_name, int priority) {
  stream_name_ = stream_name;
  priority_ = priority;
  is_absolute_stream_ = true;
}

void StreamContextScheduleable::set_stream(const std::string &stream_name, int priority) {
  stream_name_ = stream_name;
  priority_ = priority;
  is_absolute_stream_ = false;
}

std::tuple<std::string, int> StreamContextScheduleable::get_stream_name(
    std::shared_ptr<GPUResource> gpu) {
  return {is_absolute_stream_ ? stream_name_.value_or("")
                              : gpu->get_current_stream_name() + stream_name_.value_or(""),
          priority_};
}

void StreamContextScheduleable::wait_event(const std::vector<cudaEvent_t> &schedule_event,
                                           bool external) {
  HCTR_CHECK_HINT(!schedule_event_, "duplicate wait_event.");
  schedule_event_ = schedule_event;
  wait_external_ = external;
}

cudaEvent_t StreamContextScheduleable::record_done(bool external, unsigned int flags) {
  if (!completion_event_) {
    cudaEvent_t event;
    HCTR_LIB_THROW(cudaEventCreateWithFlags(&event, flags));
    completion_event_ = event;
    record_external_ = external;
  }
  HCTR_CHECK_HINT(external == record_external_, "contradictory record_done.");
  return completion_event_.value();
}

void StreamContextScheduleable::init(std::shared_ptr<GPUResource> gpu) {
  CudaDeviceContext context{gpu->get_device_id()};

  auto [current_stream_name, priority] = get_stream_name(gpu);
  StreamContext stream_context{gpu, current_stream_name, priority};
}

void StreamContextScheduleable::run(std::shared_ptr<GPUResource> gpu, bool use_graph) {
  CudaDeviceContext context{gpu->get_device_id()};

  auto [current_stream_name, priority] = get_stream_name(gpu);
  StreamContext stream_context{gpu, current_stream_name, priority};
  cudaStream_t stream = gpu->get_stream();
  if (schedule_event_.has_value()) {
    for (cudaEvent_t event : schedule_event_.value()) {
      HCTR_LIB_THROW(cudaStreamWaitEvent(
          stream, event,
          wait_external_ && use_graph ? cudaEventWaitExternal : cudaEventWaitDefault));
    }
  }
  if (workload_) workload_();
  if (completion_event_.has_value()) {
    HCTR_LIB_THROW(cudaEventRecordWithFlags(
        completion_event_.value(), stream,
        record_external_ && use_graph ? cudaEventRecordExternal : cudaEventRecordDefault));
  }
}

void GraphScheduleable::run(std::shared_ptr<GPUResource> gpu, bool use_graph) {
  if (scheduleable_list_.empty()) return;
  auto do_it = [=](cudaStream_t) {
    for (auto &scheduleable : scheduleable_list_) {
      scheduleable->run(gpu, use_graph);
    }
  };
  auto first_node = std::dynamic_pointer_cast<StreamContextScheduleable>(scheduleable_list_[0]);
  HCTR_THROW_IF(first_node == nullptr, Error_t::WrongInput,
                "first node of GraphSchedule should be StreamScheduleable");
  auto [current_stream_name, priority] = first_node->get_stream_name(gpu);
  cudaStream_t stream = gpu->get_stream(current_stream_name, priority);
  if (!use_graph) {
    do_it(stream);
    return;
  }
  if (!graph_.initialized) {
    graph_.capture(do_it, stream);
#ifdef ENABLE_MPI
#pragma omp master
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#pragma omp barrier
  }
  graph_.exec(stream);
}

Pipeline::Pipeline(const std::string &stream_name, std::shared_ptr<GPUResource> gpu_resource,
                   const std::vector<std::shared_ptr<Scheduleable>> &scheduleable_list)
    : stream_name_(stream_name),
      gpu_resource_(std::move(gpu_resource)),
      scheduleable_list_(scheduleable_list) {
  StreamContext stream_context(gpu_resource_, stream_name_);
  for (auto &scheduleable : scheduleable_list_) {
    scheduleable->init(gpu_resource_);
  }
}

void Pipeline::run() {
  StreamContext stream_context(gpu_resource_, stream_name_);
  for (auto &scheduleable : scheduleable_list_) {
    scheduleable->run(gpu_resource_, false);
  }
}

void Pipeline::run_graph() {
  StreamContext stream_context(gpu_resource_, stream_name_);
  for (auto &scheduleable : scheduleable_list_) {
    scheduleable->run(gpu_resource_, true);
  }
}

}  // namespace HugeCTR
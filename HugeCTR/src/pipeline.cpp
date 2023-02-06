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

#include <pipeline.hpp>

namespace HugeCTR {

Scheduleable::Scheduleable()
    : stream_name_(std::nullopt),
      priority_(0),
      is_absolute_stream_(false),
      schedule_event_(std::nullopt),
      wait_external_(false),
      completion_event_(std::nullopt),
      record_external_(false) {}

void Scheduleable::set_absolute_stream(const std::string &stream_name, int priority) {
  stream_name_ = stream_name;
  priority_ = priority;
  is_absolute_stream_ = true;
}

void Scheduleable::set_stream(const std::string &stream_name, int priority) {
  stream_name_ = stream_name;
  priority_ = priority;
  is_absolute_stream_ = false;
}

void Scheduleable::wait_event(const std::vector<cudaEvent_t> &schedule_event, bool external) {
  HCTR_CHECK_HINT(!schedule_event_, "duplicate wait_event.");
  schedule_event_ = schedule_event;
  wait_external_ = external;
}

cudaEvent_t Scheduleable::record_done(bool external, unsigned int flags) {
  if (!completion_event_) {
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, flags);
    completion_event_ = event;
    record_external_ = external;
  }
  HCTR_CHECK_HINT(external == record_external_, "contradictory record_done.");
  return completion_event_.value();
}

StreamContextScheduleable::StreamContextScheduleable(std::function<void()> workload)
    : workload_(workload) {}

void StreamContextScheduleable::run(std::shared_ptr<GPUResource> gpu, bool use_graph,
                                    bool prepare_resource) {
  CudaDeviceContext context{gpu->get_device_id()};

  std::string current_stream_name;
  if (is_absolute_stream_) {
    current_stream_name = stream_name_.value_or("");
  } else {
    current_stream_name = gpu->get_current_stream_name() + stream_name_.value_or("");
  }
  StreamContext stream_context{gpu, current_stream_name, priority_};
  if (!prepare_resource) {
    cudaStream_t stream = gpu->get_stream();
    if (schedule_event_.has_value()) {
      for (cudaEvent_t event : schedule_event_.value()) {
        HCTR_LIB_THROW(cudaStreamWaitEvent(
            stream, event,
            wait_external_ && use_graph ? cudaEventWaitExternal : cudaEventWaitDefault));
      }
    }
    workload_();
    if (completion_event_.has_value()) {
      HCTR_LIB_THROW(cudaEventRecordWithFlags(
          completion_event_.value(), stream,
          record_external_ && use_graph ? cudaEventRecordExternal : cudaEventRecordDefault));
    }
  }
}

Pipeline::Pipeline(const std::string &stream_name, std::shared_ptr<GPUResource> gpu_resource,
                   const std::vector<std::shared_ptr<Scheduleable>> &scheduleable_list)
    : stream_name_(stream_name),
      gpu_resource_(gpu_resource),
      scheduleable_list_(scheduleable_list) {
  StreamContext stream_context(gpu_resource_, stream_name_);
  for (auto &scheduleable : scheduleable_list_) {
    scheduleable->run(gpu_resource_, false, true);
  }
}

void Pipeline::run() {
  StreamContext stream_context(gpu_resource_, stream_name_);
  for (auto &scheduleable : scheduleable_list_) {
    scheduleable->run(gpu_resource_, false, false);
  }
}

void Pipeline::run_graph() {
  auto do_it = [this](cudaStream_t) {
    for (auto &scheduleable : scheduleable_list_) {
      scheduleable->run(gpu_resource_, true, false);
    }
  };
  StreamContext stream_context(gpu_resource_, stream_name_);
  cudaStream_t stream = gpu_resource_->get_stream();
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
}  // namespace HugeCTR

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
#include <gpu_resource.hpp>
#include <memory>
#include <vector>

namespace HugeCTR {

class GpuLearningRateScheduler {
  const float base_lr_;
  const size_t warmup_steps_;
  const size_t decay_start_;
  const size_t decay_steps_;
  const float decay_power_;
  const float end_lr_;
  size_t* step_;
  float* current_lr_;
  std::shared_ptr<GPUResource> gpu_resource_;

 public:
  GpuLearningRateScheduler(float base_lr, size_t warmup_steps, size_t decay_start,
                           size_t decay_steps, float decay_power, float end_lr,
                           const std::shared_ptr<GPUResource>& gpu_resource);
  ~GpuLearningRateScheduler();

  void update();

  float* get_learning_rate() const { return current_lr_; };

};

using GpuLearningRateSchedulers = 
std::vector<std::shared_ptr<GpuLearningRateScheduler>>;

}  // namespace HugeCTR

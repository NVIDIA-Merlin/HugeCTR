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

#include <gpu_learning_rate_scheduler.hpp>
#include <common.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

namespace {

__global__ void lr_update_kernel(float base_lr, size_t warmup_steps, size_t decay_start,
                                 size_t decay_steps, float decay_power, float end_lr,
                                 size_t* step, float* current_lr) {
  size_t step_val = *step + 1;
  *step = step_val;
  if (step_val <= warmup_steps) {
    *current_lr = step_val * base_lr / warmup_steps;
  }
  else {
    if (decay_start != 0) {
      if (step_val <= decay_start) {
        *current_lr = base_lr;
      } else if (step_val <= decay_start + decay_steps) {
        float lr_factor =
            pow(((decay_start + decay_steps - step_val) / ((float)decay_steps)), decay_power);
        *current_lr = base_lr * lr_factor > end_lr ? base_lr * lr_factor : end_lr;
      } else {
        *current_lr = end_lr;
      }
    } else {
      *current_lr = base_lr;
    }
  }
}

}  // namespace

GpuLearningRateScheduler::GpuLearningRateScheduler(
        float base_lr, size_t warmup_steps, size_t decay_start,
        size_t decay_steps, float decay_power, float end_lr,
        const std::shared_ptr<GPUResource>& gpu_resource)
    : base_lr_(base_lr),
      warmup_steps_(warmup_steps),
      decay_start_(decay_start),
      decay_steps_(decay_steps),
      decay_power_(decay_power),
      end_lr_(end_lr),
      gpu_resource_(gpu_resource) {
  if (base_lr < 0 || warmup_steps < 1 || decay_steps < 1 || decay_power < 1.0f || end_lr < 0.f) {
    CK_THROW_(Error_t::WrongInput,
              "base_lr < 0 || warmup_steps < 1 || decay_steps < 1 || decay_power < 1.0 || end_lr "
              "< 0.f");
  }

  CudaDeviceContext context(gpu_resource_->get_device_id());
  CK_CUDA_THROW_(cudaMalloc(&step_, sizeof(size_t)));
  CK_CUDA_THROW_(cudaMalloc(&current_lr_, sizeof(float)));
  initialize_array<<<1, 1, 0, gpu_resource_->get_stream()>>>(step_, 1, (size_t)0);
  lr_update_kernel<<<1, 1, 0, gpu_resource_->get_stream()>>>(
      base_lr_, warmup_steps_, decay_start_, decay_steps_, decay_power_, end_lr_, step_, current_lr_);
}

GpuLearningRateScheduler::~GpuLearningRateScheduler() {
  cudaFree(step_);
  cudaFree(current_lr_);
}

void GpuLearningRateScheduler::update() {
  CudaDeviceContext context(gpu_resource_->get_device_id());
  lr_update_kernel<<<1, 1, 0, gpu_resource_->get_stream()>>>(
      base_lr_, warmup_steps_, decay_start_, decay_steps_, decay_power_, end_lr_, step_, current_lr_);
}

}  // namespace HugeCTR

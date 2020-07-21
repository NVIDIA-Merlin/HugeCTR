/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

class LearningRateScheduler {
  const float base_lr_;
  const size_t warmup_steps_;
  const size_t decay_start_;
  const size_t decay_steps_;
  const float decay_power_;
  const float end_lr_;
  size_t step_{0};
  float current_lr_{0.f};

 public:
  // decay_start means no decay will be used.
  LearningRateScheduler(float base_lr, size_t warmup_steps = 1, size_t decay_start = 0,
                        size_t decay_steps = 1, float decay_power = 2.f, float end_lr = 0.f)
      : base_lr_(base_lr),
        warmup_steps_(warmup_steps),
        decay_start_(decay_start),
        decay_steps_(decay_steps),
        decay_power_(decay_power),
        end_lr_(end_lr) {
    if (base_lr < 0 || warmup_steps < 1 || decay_steps < 1 || decay_power < 1.0f || end_lr < 0.f) {
      CK_THROW_(Error_t::WrongInput,
                "base_lr < 0 || warmup_steps < 1 || decay_steps < 1 || decay_power < 1.0 || end_lr "
                "< 0.f");
    }
  }

  float get_next() {
    step_++;
    if (step_ <= warmup_steps_) {
      current_lr_ = step_ * base_lr_ / warmup_steps_;
    } else {
      if (decay_start_ != 0) {
        if (step_ <= decay_start_) {
          current_lr_ = base_lr_;
        } else {
          float lr_factor =
              pow(((decay_start_ + decay_steps_ - step_) / ((float)decay_steps_)), decay_power_);
          current_lr_ = base_lr_ * lr_factor > end_lr_ ? base_lr_ * lr_factor : end_lr_;
        }
      } else {
        current_lr_ = base_lr_;
      }
    }
    return current_lr_;
  }

  float get_lr() const { return current_lr_; }

  size_t get_step() const { return step_; }
};

/**
 * @brief Base class for all optimizers
 */
class Optimizer {
 public:
  /**
   * Helper to create a speicifed Optimizer object
   */
  template <typename T>
  static std::unique_ptr<Optimizer> Create(const OptParams<T>& params,
                                           const GeneralBufferPtr<float>& weight_main,
                                           const GeneralBufferPtr<float>& wgrad,
                                           const GeneralBufferPtr<__half>& wgrad_half,
                                           bool mixed_precision, const float scaler, int device_id);

  /**
   * Constructor of Optimizer.
   * @param weight_main weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate
   */
  Optimizer(const std::shared_ptr<GeneralBuffer<float>>& weight_main,
            const GeneralBufferPtr<float>& fp32_wgrad, const GeneralBufferPtr<__half>& fp16_wgrad,
            bool mixed_precision, int device_id, float learning_rate, float scaler)
      : weight_main_(weight_main),
        fp32_wgrad_(fp32_wgrad),
        fp16_wgrad_(fp16_wgrad),
        mixed_precision_(mixed_precision),
        device_id_(device_id),
        lr_(learning_rate),
        scaler_(scaler) {
    if (mixed_precision) {
      if (weight_main->get_num_elements() != fp16_wgrad->get_num_elements()) {
        CK_THROW_(Error_t::WrongInput,
                  "fp32_weight->get_num_elements() != fp16_wgrad->get_num_elements()");
      }
    } else {
      if (weight_main->get_num_elements() != fp32_wgrad->get_num_elements()) {
        CK_THROW_(Error_t::WrongInput,
                  "fp32_weight->get_num_elements() != fp32_wgrad->get_num_elements()");
      }
    }
    if (lr_ <= 0.) {
      CK_THROW_(Error_t::WrongInput, "lr <= 0");
    }
  }

  virtual ~Optimizer() {}

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  virtual void update(cudaStream_t stream) = 0;

  /**
   * update the learning rate
   * @param lr the learning rate
   */
  void set_learning_rate(float lr) {
    if (lr <= 0) {
      CK_THROW_(Error_t::WrongInput, "lr <= 0");
    }
    lr_ = lr;
  }

 protected:
  GeneralBufferPtr<float> weight_main_;
  GeneralBufferPtr<float> fp32_wgrad_;
  GeneralBufferPtr<__half> fp16_wgrad_;
  bool mixed_precision_;
  int device_id_;
  float lr_;  // learning rate
  const float scaler_;
};

}  // namespace HugeCTR

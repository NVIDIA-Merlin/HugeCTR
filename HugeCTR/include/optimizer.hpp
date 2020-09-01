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
#include <common.hpp>
#include <general_buffer2.hpp>
#include <gpu_resource.hpp>

namespace HugeCTR {

template <typename T>
struct AdamOptHyperParams {  // TODO: move to optimizer
  uint64_t times = 0;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-7f;
  T* m_ptr = nullptr;
  T* v_ptr = nullptr;
};

template <typename T>
struct MomentumSGDOptHyperParams {
  float factor = 0.1f;
  T* momentum_ptr = nullptr;
};

template <typename T>
struct NesterovOptHyperParams {
  float mu = 0.9f;
  T* accm_ptr = nullptr;
};

struct SGDOptHyperParams {
  bool atomic_update = false;
};

// TODO: use union type should be better ???
template <typename TypeEmbeddingComp>
struct OptHyperParams {
  AdamOptHyperParams<TypeEmbeddingComp> adam;
  MomentumSGDOptHyperParams<TypeEmbeddingComp> momentum;
  NesterovOptHyperParams<TypeEmbeddingComp> nesterov;
  SGDOptHyperParams sgd;
};

template <typename TypeEmbeddingComp>
struct OptParams {
  Optimizer_t optimizer;
  float lr;
  OptHyperParams<TypeEmbeddingComp> hyperparams;
  bool global_update;
  float scaler;
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
  static std::unique_ptr<Optimizer> Create(
      const OptParams<T>& params, const Tensor2<float>& weight_main, const Tensor2<float>& wgrad,
      const Tensor2<__half>& wgrad_half, bool mixed_precision, const float scaler,
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& buff,
      const std::shared_ptr<GPUResource>& gpu_resource);

  /**
   * Constructor of Optimizer.
   * @param weight_main weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate
   */
  Optimizer(const Tensor2<float>& weight_main, const Tensor2<float>& fp32_wgrad,
            const Tensor2<__half>& fp16_wgrad, bool mixed_precision,
            const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate, float scaler)
      : weight_main_(weight_main),
        fp32_wgrad_(fp32_wgrad),
        fp16_wgrad_(fp16_wgrad),
        mixed_precision_(mixed_precision),
        gpu_resource_(gpu_resource),
        lr_(learning_rate),
        scaler_(scaler) {
    if (mixed_precision) {
      if (weight_main.get_num_elements() != fp16_wgrad.get_num_elements()) {
        CK_THROW_(Error_t::WrongInput,
                  "fp32_weight->get_num_elements() != fp16_wgrad->get_num_elements()");
      }
    } else {
      if (weight_main.get_num_elements() != fp32_wgrad.get_num_elements()) {
        CK_THROW_(Error_t::WrongInput,
                  "fp32_weight->get_num_elements() != fp32_wgrad->get_num_elements()");
      }
    }
    if (lr_ <= 0.) {
      CK_THROW_(Error_t::WrongInput, "lr <= 0");
    }
  }

  virtual ~Optimizer() {}

  virtual void initialize() {}

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  virtual void update() = 0;

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
  Tensor2<float> weight_main_;
  Tensor2<float> fp32_wgrad_;
  Tensor2<__half> fp16_wgrad_;
  bool mixed_precision_;
  std::shared_ptr<GPUResource> gpu_resource_;
  float lr_;  // learning rate
  const float scaler_;

  int get_device_id() const { return gpu_resource_->get_device_id(); }
};

}  // namespace HugeCTR

/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

/**
 * @brief Base class for all optimizers
 */
class Optimizer {
 public:
  /**
   * Constructor of Optimizer.
   * @param weight weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate
   */
  Optimizer(const std::shared_ptr<GeneralBuffer<float>>& weight,
            const std::shared_ptr<GeneralBuffer<float>>& wgrad, int device_id, float learning_rate, 
	    float scaler)
    : device_id_(device_id), weight_(weight), wgrad_(wgrad), lr_(learning_rate), scaler_(scaler) {
    try {
      if (weight_->get_size() != wgrad_->get_size()) {
        CK_THROW_(Error_t::WrongInput, "weight_.get_size() != wgrad_.get_size()");
      }
      if (lr_ <= 0) {
        CK_THROW_(Error_t::WrongInput, "lr <= 0");
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  virtual void update(cudaStream_t stream) = 0;
  void set_learning_rate(float lr) {
    if (lr <= 0) {
      CK_THROW_(Error_t::WrongInput, "lr <= 0");
    }
    lr_ = lr;
  }

  /**
   * destructor of Optimizer
   */
  virtual ~Optimizer() {}

 protected:
  int device_id_;
  std::shared_ptr<GeneralBuffer<float>> weight_;
  std::shared_ptr<GeneralBuffer<float>> wgrad_;
  float lr_;  // learning rate
  const float scaler_;
};

}  // namespace HugeCTR

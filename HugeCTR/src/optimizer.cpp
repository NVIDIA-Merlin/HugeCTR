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

#include "HugeCTR/include/optimizer.hpp"

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/optimizers/adam_optimizer.hpp"
#include "HugeCTR/include/optimizers/momentum_sgd.hpp"
#include "HugeCTR/include/optimizers/nesterov_optimizer.hpp"
#include "HugeCTR/include/optimizers/sgd_optimizer.hpp"

#include <type_traits>

namespace HugeCTR {

template <typename T>
std::unique_ptr<Optimizer>
Optimizer::Create(const OptParams<float>& params,
                  const std::shared_ptr<GeneralBuffer<float>>& weight_main,
                  const std::shared_ptr<GeneralBuffer<T>>& wgrad,
                  const std::shared_ptr<GeneralBuffer<T>>& weight_sub,
                  const float scaler,
                  int device_id) {
  if (std::is_same<T, __half>::value == false && weight_sub != nullptr) {
    assert(!"Error: weight_sub is only valid when T == __half");
  }

  std::unique_ptr<Optimizer> ret;

  switch (params.optimizer) {
    case Optimizer_t::Adam: {
      auto alpha = params.lr;
      auto beta1 = params.hyperparams.adam.beta1;
      auto beta2 = params.hyperparams.adam.beta2;
      auto epsilon = params.hyperparams.adam.epsilon;
      ret.reset(new AdamOptimizer<T>(weight_main, wgrad, weight_sub,
                                     device_id,
                                     alpha,
                                     beta1, beta2,
                                     epsilon, scaler));
      break;
    }
    case Optimizer_t::MomentumSGD: {
      auto learning_rate = params.lr;
      auto momentum_factor = params.hyperparams.momentum.factor;
      ret.reset(new MomentumSGD<T>(weight_main, wgrad, weight_sub,
                                   device_id,
                                   learning_rate,
                                   momentum_factor, scaler));
      break;
    }
    case Optimizer_t::Nesterov: {
      auto learning_rate = params.lr;
      auto momentum_factor = params.hyperparams.nesterov.mu;
      ret.reset(new NesterovOptimizer<T>(weight_main, wgrad, weight_sub,
                                         device_id,
                                         learning_rate, momentum_factor,
                                         scaler));
      break;
    }
    case Optimizer_t::SGD: {
      auto learning_rate = params.lr;
      ret.reset(new SgdOptimizer<T>(weight_main, wgrad, weight_sub,
                                    device_id,
                                    learning_rate, scaler));
      break;
    }
    default:
      assert(!"Error: no such optimizer && should never get here!");
  }
  return ret;
}

template std::unique_ptr<Optimizer>
Optimizer::Create<float>(const OptParams<float>& params,
                         const std::shared_ptr<GeneralBuffer<float>>& weight_main,
                         const std::shared_ptr<GeneralBuffer<float>>& wgrad,
                         const std::shared_ptr<GeneralBuffer<float>>& weight_sub,
                         const float scaler,
                         int device_id);

template std::unique_ptr<Optimizer>
Optimizer::Create<__half>(const OptParams<float>& params,
                          const std::shared_ptr<GeneralBuffer<float>>& weight_main,
                          const std::shared_ptr<GeneralBuffer<__half>>& wgrad,
                          const std::shared_ptr<GeneralBuffer<__half>>& weight_sub,
                          const float scaler,
                          int device_id);

} // end namespace HugeCTR

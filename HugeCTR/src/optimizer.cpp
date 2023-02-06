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

#include <optimizer.hpp>
#include <optimizers/adagrad_optimizer.hpp>
#include <optimizers/adam_optimizer.hpp>
#include <optimizers/ftrl_optimizer.hpp>
#include <optimizers/momentum_sgd_optimizer.hpp>
#include <optimizers/nesterov_optimizer.hpp>
#include <optimizers/sgd_optimizer.hpp>
#include <type_traits>

namespace HugeCTR {

OptParamsPy::OptParamsPy() : initialized(false) {}

OptParamsPy::OptParamsPy(Optimizer_t optimizer_type, Update_t update_t,
                         OptHyperParams opt_hyper_params)
    : optimizer(optimizer_type),
      update_type(update_t),
      hyperparams(opt_hyper_params),
      initialized(true) {}

template <typename T>
std::unique_ptr<Optimizer> Optimizer::Create(const OptParams& params,
                                             const Tensor2<float>& weight_main,
                                             const Tensor2<__half>& weight_main_half,
                                             const Tensor2<T>& wgrad, const float scaler,
                                             const std::shared_ptr<BufferBlock2<float>>& opt_buff,
                                             const std::shared_ptr<GPUResource>& gpu_resource,
                                             bool use_mixed_precision) {
  std::unique_ptr<Optimizer> ret;

  switch (params.optimizer) {
    case Optimizer_t::Ftrl: {
      auto lr = params.lr;
      auto beta = params.hyperparams.ftrl.beta;
      auto lambda1 = params.hyperparams.ftrl.lambda1;
      auto lambda2 = params.hyperparams.ftrl.lambda2;
      ret = std::make_unique<FtrlOptimizer<T>>(weight_main, wgrad, opt_buff, gpu_resource, lr, beta,
                                               lambda1, lambda2, scaler);
    } break;

    case Optimizer_t::Adam: {
      auto lr = params.lr;
      auto beta1 = params.hyperparams.adam.beta1;
      auto beta2 = params.hyperparams.adam.beta2;
      auto epsilon = params.hyperparams.adam.epsilon;
      ret = std::make_unique<AdamOptimizer<T>>(weight_main, wgrad, opt_buff, gpu_resource, lr,
                                               beta1, beta2, epsilon, scaler);
    } break;

    case Optimizer_t::AdaGrad: {
      auto lr = params.lr;
      auto initial_accu_value = params.hyperparams.adagrad.initial_accu_value;
      auto epsilon = params.hyperparams.adagrad.epsilon;
      ret = std::make_unique<AdaGradOptimizer<T>>(weight_main, wgrad, opt_buff, gpu_resource, lr,
                                                  initial_accu_value, epsilon, scaler);
    } break;

    case Optimizer_t::MomentumSGD: {
      auto learning_rate = params.lr;
      auto momentum_factor = params.hyperparams.momentum.factor;
      ret = std::make_unique<MomentumSGDOptimizer<T>>(weight_main, wgrad, opt_buff, gpu_resource,
                                                      learning_rate, momentum_factor, scaler);
    } break;

    case Optimizer_t::Nesterov: {
      auto learning_rate = params.lr;
      auto momentum_factor = params.hyperparams.nesterov.mu;
      ret = std::make_unique<NesterovOptimizer<T>>(weight_main, wgrad, opt_buff, gpu_resource,
                                                   learning_rate, momentum_factor, scaler);
    } break;

    case Optimizer_t::SGD: {
      auto learning_rate = params.lr;
      ret = std::make_unique<SGDOptimizer<T>>(weight_main, weight_main_half, wgrad, gpu_resource,
                                              learning_rate, scaler, use_mixed_precision);
    } break;

    default:
      HCTR_OWN_THROW(Error_t::WrongInput, "No such optimizer && should never get here!");
  }

  return ret;
}

template std::unique_ptr<Optimizer> Optimizer::Create<float>(
    const OptParams& params, const Tensor2<float>& weight_main,
    const Tensor2<__half>& weight_main_half, const Tensor2<float>& wgrad, const float scaler,
    const std::shared_ptr<BufferBlock2<float>>& opt_buff,
    const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision);

template std::unique_ptr<Optimizer> Optimizer::Create<__half>(
    const OptParams& params, const Tensor2<float>& weight_main,
    const Tensor2<__half>& weight_main_half, const Tensor2<__half>& wgrad, const float scaler,
    const std::shared_ptr<BufferBlock2<float>>& opt_buff,
    const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision);

}  // end namespace HugeCTR

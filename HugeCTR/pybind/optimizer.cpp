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

#include <HugeCTR/pybind/optimizer.hpp>

namespace HugeCTR {

OptParamsBase::~OptParamsBase() {}
OptParamsBase::OptParamsBase(bool use_mixed_precision) : use_mixed_precision(use_mixed_precision) {}

template <typename TypeEmbeddingComp>
OptParamsPy<TypeEmbeddingComp>::OptParamsPy(Optimizer_t optimizer_type, float learning_rate, 
                                          OptHyperParams<TypeEmbeddingComp> opt_hyper_params, 
                                          Update_t update_t, size_t warmup_steps, size_t decay_start,
                                          size_t decay_steps, float decay_power, float end_lr, bool use_mixed_precision)
  : OptParamsBase(use_mixed_precision), optimizer(optimizer_type), lr(learning_rate), hyperparams(opt_hyper_params), update_type(update_t),
    warmup_steps(warmup_steps), decay_start(decay_start), decay_steps(decay_steps),
    decay_power(decay_power), end_lr(end_lr) {}

template class OptParamsPy<__half>;
template class OptParamsPy<float>;

} // namespace HugeCTR
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
#include <HugeCTR/include/optimizer.hpp>

namespace HugeCTR {

class OptParamsBase {
public:
  OptParamsBase(bool use_mixed_precision);
  virtual ~OptParamsBase();
  bool use_mixed_precision;
};

template <typename TypeEmbeddingComp>
class OptParamsPy : public OptParamsBase {
public:
  Optimizer_t optimizer;
  float lr;
  OptHyperParams<TypeEmbeddingComp> hyperparams;
  Update_t update_type;
  size_t warmup_steps;
  size_t decay_start;
  size_t decay_steps;
  float decay_power;
  float end_lr;
  OptParamsPy(Optimizer_t optimizer_type, float learning_rate, 
            OptHyperParams<TypeEmbeddingComp> opt_hyper_params, Update_t update_t,
            size_t warmup_steps, size_t decay_start,
            size_t decay_steps, float decay_power, float end_lr, bool use_mixed_precision);
};

} //namespace HugeCTR
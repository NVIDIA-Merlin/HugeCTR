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

struct AdamOptHyperParamsPy {
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-7f;
};

struct MomentumSGDOptHyperParamsPy {
  float factor = 0.1f;
};

struct NesterovOptHyperParamsPy {
  float mu = 0.9f;
};

struct SGDOptHyperParamsPy {
  bool atomic_update = false;
};

struct OptHyperParamsPy {
  AdamOptHyperParamsPy adam;
  MomentumSGDOptHyperParamsPy momentum;
  NesterovOptHyperParamsPy nesterov;
  SGDOptHyperParamsPy sgd;
};

class OptParamsPy {
public:
  Optimizer_t optimizer;
  Update_t update_type;
  OptHyperParamsPy hyperparams;
  OptParamsPy(Optimizer_t optimizer_type, Update_t update_t, OptHyperParamsPy opt_hyper_params);
  OptParamsPy();
};

} //namespace HugeCTR
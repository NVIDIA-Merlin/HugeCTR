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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <HugeCTR/include/optimizer.hpp>


namespace HugeCTR {

namespace python_lib {

std::shared_ptr<OptParamsPy> CreateOptimizer(Optimizer_t optimizer_type,
                                            Update_t update_type,
                                            float beta1, float beta2, float epsilon,
                                            float initial_accu_value, float momentum_factor,
                                            bool atomic_update) {
  std::shared_ptr<OptParamsPy> opt_params;
  OptHyperParams opt_hyper_params;
  opt_hyper_params.adam.beta1 = beta1;
  opt_hyper_params.adam.beta2 = beta2;
  opt_hyper_params.adam.epsilon = epsilon;
  opt_hyper_params.adagrad.initial_accu_value = initial_accu_value;
  opt_hyper_params.adagrad.epsilon = epsilon;
  opt_hyper_params.momentum.factor = momentum_factor;
  opt_hyper_params.nesterov.mu = momentum_factor;
  opt_hyper_params.sgd.atomic_update = atomic_update;
  opt_params.reset(new OptParamsPy(optimizer_type, update_type, opt_hyper_params));
  return opt_params;
}

void OptimizerPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::OptParamsPy, std::shared_ptr<HugeCTR::OptParamsPy>>(m, "OptParamsPy");
  m.def("CreateOptimizer", &HugeCTR::python_lib::CreateOptimizer,
    pybind11::arg("optimizer_type") = HugeCTR::Optimizer_t::Adam,
    pybind11::arg("update_type") = HugeCTR::Update_t::Global,
    pybind11::arg("beta1") = 0.9,
    pybind11::arg("beta2") = 0.999,
    pybind11::arg("epsilon") = 0.0000001,
    pybind11::arg("initial_accu_value") = 0.f,
    pybind11::arg("momentum_factor") = 0.0,
    pybind11::arg("atomic_update") = true);
}

} // namespace python_lib

} // namespace HugeCTR

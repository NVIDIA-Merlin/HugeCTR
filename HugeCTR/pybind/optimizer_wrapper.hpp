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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <HugeCTR/pybind/optimizer.hpp>


namespace HugeCTR {

namespace python_lib {

std::shared_ptr<OptParamsBase> CreateOptimizer(Optimizer_t optimizer_type,
                                              Update_t update_type, float learning_rate,
                                              size_t warmup_steps, size_t decay_start,
                                              size_t decay_steps, float decay_power, float end_lr,
                                              float beta1, float beta2, float epsilon,
                                              float momentum_factor, bool atomic_update, bool use_mixed_precision) {
  std::shared_ptr<OptParamsBase> opt_params;
  if (use_mixed_precision) {
    OptHyperParams<__half> opt_hyper_params;
    opt_hyper_params.adam.beta1 = beta1;
    opt_hyper_params.adam.beta2 = beta2;
    opt_hyper_params.adam.epsilon = epsilon;
    opt_hyper_params.momentum.factor = momentum_factor;
    opt_hyper_params.nesterov.mu = momentum_factor;
    opt_hyper_params.sgd.atomic_update = atomic_update;
    opt_params.reset(new OptParamsPy<__half>(optimizer_type, learning_rate, opt_hyper_params, update_type,
                                            warmup_steps, decay_start, decay_steps, decay_power, end_lr, use_mixed_precision));
  } else {
    OptHyperParams<float> opt_hyper_params;
    opt_hyper_params.adam.beta1 = beta1;
    opt_hyper_params.adam.beta2 = beta2;
    opt_hyper_params.adam.epsilon = epsilon;
    opt_hyper_params.momentum.factor = momentum_factor;
    opt_hyper_params.nesterov.mu = momentum_factor;
    opt_hyper_params.sgd.atomic_update = atomic_update;
    opt_params.reset(new OptParamsPy<float>(optimizer_type, learning_rate, opt_hyper_params, update_type,
                                            warmup_steps, decay_start, decay_steps, decay_power, end_lr, use_mixed_precision));
  }
  return opt_params;
}

void OptimizerPybind(pybind11::module& m) {
  pybind11::module opt = m.def_submodule("optimizer", "optimizer submodule of hugectr");
  pybind11::class_<HugeCTR::OptParamsBase, std::shared_ptr<HugeCTR::OptParamsBase>>(opt, "OptParamsBase")
    .def_readonly("use_mixed_precision", &HugeCTR::OptParamsBase::use_mixed_precision);
  opt.def("CreateOptimizer", &HugeCTR::python_lib::CreateOptimizer,
    pybind11::arg("optimizer_type") = HugeCTR::Optimizer_t::Adam,
    pybind11::arg("update_type") = HugeCTR::Update_t::Global,
    pybind11::arg("learning_rate") = 0.001,
    pybind11::arg("warmup_steps") = 1, 
    pybind11::arg("decay_start") = 0,
    pybind11::arg("decay_steps") = 1, 
    pybind11::arg("decay_power") = 2.f,
    pybind11::arg("end_lr") = 0.f,
    pybind11::arg("beta1") = 0.9,
    pybind11::arg("beta2") = 0.999,
    pybind11::arg("epsilon") = 0.0000001,
    pybind11::arg("momentum_factor") = 0.0,
    pybind11::arg("atomic_update") = true,
    pybind11::arg("use_mixed_precision"));
}

} // namespace python_lib

} // namespace HugeCTR

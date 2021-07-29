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

#include <pybind11/pybind11.h>

#include <HugeCTR/pybind/common_wrapper.hpp>
#include <HugeCTR/pybind/data_reader_wrapper.hpp>
#include <HugeCTR/pybind/inference_wrapper.hpp>
#include <HugeCTR/pybind/learning_rate_scheduler_wrapper.hpp>
#include <HugeCTR/pybind/model_oversubscriber_wrapper.hpp>
#include <HugeCTR/pybind/model_wrapper.hpp>
#include <HugeCTR/pybind/optimizer_wrapper.hpp>
#include <HugeCTR/pybind/solver_wrapper.hpp>
using namespace HugeCTR::python_lib;

PYBIND11_MODULE(hugectr, m) {
  m.doc() = "hugectr python interface";
  CommonPybind(m);
  SolverPybind(m);
  DataReaderPybind(m);
  ModelOversubscriberPybind(m);
  LearningRateSchedulerPybind(m);
  OptimizerPybind(m);
  ModelPybind(m);
  InferencePybind(m);
}

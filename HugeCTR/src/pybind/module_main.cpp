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

#include <pybind11/pybind11.h>

#if defined(ENABLE_INFERENCE)
#include <pybind/hps_wrapper.hpp>
#endif

#include <pybind/common_wrapper.hpp>
#include <pybind/data_generator_wrapper.hpp>
#include <pybind/data_reader_wrapper.hpp>
#include <pybind/data_source_wrapper.hpp>
#include <pybind/embedding_collection_wrapper.hpp>
#include <pybind/learning_rate_scheduler_wrapper.hpp>
#include <pybind/model_wrapper.hpp>
#include <pybind/optimizer_wrapper.hpp>
#include <pybind/solver_wrapper.hpp>
#include <pybind/training_callback_wrapper.hpp>

using namespace HugeCTR::python_lib;

PYBIND11_MODULE(hugectr, m) {
  m.doc() = "hugectr python interface";
  CommonPybind(m);
  DataGeneratorPybind(m);
  DataSourcePybind(m);
  SolverPybind(m);
  DataReaderPybind(m);
  LearningRateSchedulerPybind(m);
  OptimizerPybind(m);
  ModelPybind(m);
  EmbeddingCollectionPybind(m);
  TrainingCallbackPybind(m);
}

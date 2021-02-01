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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <HugeCTR/include/parser.hpp>

namespace HugeCTR {

namespace python_lib {

std::unique_ptr<SolverParser> solver_parser_helper(
    unsigned long long seed, int max_eval_batches, int batchsize_eval,
    int batchsize, std::string model_file, std::string dense_opt_states_file,
    std::vector<std::string> embedding_files, std::vector<std::string> sparse_opt_states_files,
    std::vector<std::vector<int>> vvgpu,
    bool use_mixed_precision, bool enable_tf32_compute, float scaler, bool i64_input_key,
    bool use_algorithm_search, bool use_cuda_graph, bool repeat_dataset,
    int max_iter, int num_epochs, int display, int snapshot, int eval_interval,
    bool use_model_oversubscriber, std::string temp_embedding_dir) {
  std::unique_ptr<SolverParser> solver_config(new SolverParser());
  solver_config->seed = seed;
  solver_config->max_eval_batches = max_eval_batches;
  solver_config->batchsize_eval = batchsize_eval;
  solver_config->batchsize = batchsize;
  solver_config->model_file = model_file;
  solver_config->dense_opt_states_file = dense_opt_states_file;
  solver_config->embedding_files.assign(embedding_files.begin(), embedding_files.end());
  solver_config->sparse_opt_states_files.assign(sparse_opt_states_files.begin(), sparse_opt_states_files.end());
  solver_config->vvgpu.assign(vvgpu.begin(), vvgpu.end());
  solver_config->use_mixed_precision = use_mixed_precision;
  solver_config->enable_tf32_compute = enable_tf32_compute;
  solver_config->scaler = scaler;
  solver_config->i64_input_key = i64_input_key;
  solver_config->use_algorithm_search = use_algorithm_search;
  solver_config->use_cuda_graph = use_cuda_graph;
  solver_config->lr_policy = LrPolicy_t::fixed;
  solver_config->use_model_oversubscriber = use_model_oversubscriber;
  solver_config->temp_embedding_dir = temp_embedding_dir;
  solver_config->display = display;
  solver_config->max_iter = repeat_dataset?(max_iter>0?max_iter:10000):0;
  solver_config->num_epochs = repeat_dataset?0:(num_epochs>0?num_epochs:1);
  solver_config->snapshot = snapshot;
  solver_config->eval_interval = eval_interval;
  solver_config->snapshot_prefix = "./";
  solver_config->metrics_spec[metrics::Type::AUC] = 1.f;
  if (repeat_dataset && (solver_config->num_epochs > 0 || solver_config->max_iter <= 0)) {
    CK_THROW_(Error_t::WrongInput, "Require num_epochs==0 and max_iter>0 under non-epoch mode");
  }
  if (!repeat_dataset && (solver_config->num_epochs <= 0 || solver_config->max_iter > 0)) {
    CK_THROW_(Error_t::WrongInput, "Require num_epochs>0 and max_iter==0 under epoch mode");
  }
  return solver_config;
}

void SolverParserPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::SolverParser, std::unique_ptr<HugeCTR::SolverParser>>(m, "SolverParser")
      .def(pybind11::init<const std::string&>(), pybind11::arg("file"))
      .def(pybind11::init<>())
      .def_readonly("seed", &HugeCTR::SolverParser::seed)
      .def_readonly("max_eval_batches", &HugeCTR::SolverParser::max_eval_batches)
      .def_readonly("batchsize_eval", &HugeCTR::SolverParser::batchsize_eval)
      .def_readonly("batchsize", &HugeCTR::SolverParser::batchsize)
      .def_readonly("model_file", &HugeCTR::SolverParser::model_file)
      .def_readonly("dense_opt_states_file", &HugeCTR::SolverParser::dense_opt_states_file)
      .def_readonly("embedding_files", &HugeCTR::SolverParser::embedding_files)
      .def_readonly("sparse_opt_states_files", &HugeCTR::SolverParser::sparse_opt_states_files)
      .def_readonly("vvgpu", &HugeCTR::SolverParser::vvgpu)
      .def_readonly("use_mixed_precision", &HugeCTR::SolverParser::use_mixed_precision)
      .def_readonly("enable_tf32_compute", &HugeCTR::SolverParser::enable_tf32_compute)
      .def_readonly("scaler", &HugeCTR::SolverParser::scaler)
      .def_readonly("i64_input_key", &HugeCTR::SolverParser::i64_input_key)
      .def_readonly("use_algorithm_search", &HugeCTR::SolverParser::use_algorithm_search)
      .def_readonly("use_cuda_graph", &HugeCTR::SolverParser::use_cuda_graph);
  m.def("solver_parser_helper", &HugeCTR::python_lib::solver_parser_helper,
       pybind11::arg("seed") = 0,
       pybind11::arg("max_eval_batches") = 100,
       pybind11::arg("batchsize_eval") = 2048,
       pybind11::arg("batchsize") = 2048,
       pybind11::arg("model_file") = "",
       pybind11::arg("dense_opt_states_file") = "",
       pybind11::arg("embedding_files") = std::vector<std::string>(),
       pybind11::arg("sparse_opt_states_file") = std::vector<std::string>(),
       pybind11::arg("vvgpu") = std::vector<std::vector<int>>(1, std::vector<int>(1, 0)),
       pybind11::arg("use_mixed_precision") = false,
       pybind11::arg("enable_tf32_compute") = false,
       pybind11::arg("scaler") = 1.f,
       pybind11::arg("i64_input_key") = false,
       pybind11::arg("use_algorithm_search") = true,
       pybind11::arg("use_cuda_graph") = true,
       pybind11::arg("repeat_dataset") = true,
       pybind11::arg("max_iter") = 0,
       pybind11::arg("num_epochs") = 0,
       pybind11::arg("display") = 200,
       pybind11::arg("snapshot") = 10000,
       pybind11::arg("eval_interval") = 1000,
       pybind11::arg("use_model_oversubscriber") = false,
       pybind11::arg("temp_embedding_dir") = "./");
}

}  // namespace python_lib

}  // namespace HugeCTR

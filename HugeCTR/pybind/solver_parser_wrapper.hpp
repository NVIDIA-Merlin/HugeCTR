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

std::unique_ptr<SolverParser> solver_parser_helper(unsigned long long seed,
		            LrPolicy_t lr_policy,
			    int eval_interval,
			    int eval_batches,
			    int batchsize_eval,
		            int batchsize,
			    std::string model_file,
			    std::vector<std::string> embedding_files,
			    std::vector<std::vector<int>> vvgpu,
			    bool use_mixed_precision,
			    float scaler,
			    std::map<metrics::Type, float> metrics_spec,
			    bool i64_input_key,
			    bool use_algorithm_search,
			    bool use_cuda_graph
			    ) {
  std::unique_ptr<SolverParser> solver_config(new SolverParser());
  solver_config->seed = seed;
  solver_config->lr_policy = lr_policy;
  solver_config->eval_interval = eval_interval;
  solver_config->eval_batches = eval_batches;
  solver_config->batchsize_eval = batchsize_eval;
  solver_config->batchsize = batchsize;
  solver_config->model_file = model_file;
  solver_config->embedding_files.assign(embedding_files.begin(), embedding_files.end());
  solver_config->vvgpu.assign(vvgpu.begin(), vvgpu.end());
  solver_config->use_mixed_precision = use_mixed_precision;
  solver_config->scaler = scaler;
  solver_config->metrics_spec = metrics_spec;
  solver_config->i64_input_key = i64_input_key;
  solver_config->use_algorithm_search = use_algorithm_search;
  solver_config->use_cuda_graph = use_cuda_graph;
  return solver_config;
}
 
void SolverParserPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::SolverParser, std::unique_ptr<HugeCTR::SolverParser>>(m, "SolverParser")
    .def(pybind11::init<const std::string&>(), pybind11::arg("file"))
    .def(pybind11::init<>())
    .def_readonly("seed", &HugeCTR::SolverParser::seed)
    .def_readonly("lr_policy", &HugeCTR::SolverParser::lr_policy)
    .def_readonly("eval_interval", &HugeCTR::SolverParser::eval_interval)
    .def_readonly("eval_batches", &HugeCTR::SolverParser::eval_batches)
    .def_readonly("batchsize_eval", &HugeCTR::SolverParser::batchsize_eval)
    .def_readonly("batchsize", &HugeCTR::SolverParser::batchsize)
    .def_readonly("model_file", &HugeCTR::SolverParser::model_file)
    .def_readonly("embedding_files", &HugeCTR::SolverParser::embedding_files)
    .def_readonly("vvgpu", &HugeCTR::SolverParser::vvgpu)
    .def_readonly("use_mixed_precision", &HugeCTR::SolverParser::use_mixed_precision)
    .def_readonly("scaler", &HugeCTR::SolverParser::scaler)
    .def_readonly("metrics_spec", &HugeCTR::SolverParser::metrics_spec)
    .def_readonly("i64_input_key", &HugeCTR::SolverParser::i64_input_key)
    .def_readonly("use_algorithm_search", &HugeCTR::SolverParser::use_algorithm_search)
    .def_readonly("use_cuda_graph", &HugeCTR::SolverParser::use_cuda_graph);
  std::map<metrics::Type, float> metrics = {{metrics::Type::AUC, 0.8025}};
  m.def("solver_parser_helper", &HugeCTR::python_lib::solver_parser_helper,
		                  pybind11::arg("seed") = 0,
				  pybind11::arg("lr_policy") = LrPolicy_t::fixed,
				  pybind11::arg("eval_interval") = 1000,
				  pybind11::arg("eval_batches") = 100, 
				  pybind11::arg("batchsize_eval") = 16384,
				  pybind11::arg("batchsize") = 16384,
				  pybind11::arg("model_file") = "",
				  pybind11::arg("embedding_files") = std::vector<std::string>(),
				  pybind11::arg("vvgpu") = std::vector<std::vector<int>>(1, std::vector<int>(1,0)),
				  pybind11::arg("use_mixed_precision") = false,
				  pybind11::arg("scaler") = 1.f,
				  pybind11::arg("metrics_spec") = metrics,
				  pybind11::arg("i64_input_key") = false,
				  pybind11::arg("use_algorithm_search") = true,
				  pybind11::arg("use_cuda_graph") = true);
}

}  // namespace python_lib

}  // namespace HugeCTR





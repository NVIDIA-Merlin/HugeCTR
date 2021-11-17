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
#include <pybind11/stl.h>

#include <HugeCTR/include/parser.hpp>

namespace HugeCTR {

namespace python_lib {

std::unique_ptr<Solver> CreateSolver(
    unsigned long long seed, LrPolicy_t lr_policy, float lr, size_t warmup_steps,
    size_t decay_start, size_t decay_steps, float decay_power, float end_lr, int max_eval_batches,
    int batchsize_eval, int batchsize, std::vector<std::vector<int>> vvgpu, bool repeat_dataset,
    bool use_mixed_precision, bool enable_tf32_compute, float scaler,
    std::map<metrics::Type, float> metrics_spec, bool i64_input_key, bool use_algorithm_search,
    bool use_cuda_graph, bool async_mlp_wgrad, bool gen_loss_summary, bool overlap_ar_a2a, DeviceMap::Layout device_layout,
    bool use_holistic_cuda_graph, bool use_overlapped_pipeline, AllReduceAlgo all_reduce_algo,
    bool grouped_all_reduce, size_t num_iterations_statistics, bool is_dlrm) {
  if (use_mixed_precision && enable_tf32_compute) {
    CK_THROW_(Error_t::WrongInput,
              "use_mixed_precision and enable_tf32_compute cannot be true at the same time");
  }
  if (use_mixed_precision && scaler != 128 && scaler != 256 && scaler != 512 && scaler != 1024) {
    CK_THROW_(Error_t::WrongInput,
              "Scaler of mixed precision training should be either 128/256/512/1024");
  }
  if (!is_dlrm && use_holistic_cuda_graph) {
    CK_THROW_(Error_t::WrongInput, "Holistic cuda graph is restricted to DLRM use");
  }
  if (!is_dlrm && use_overlapped_pipeline) {
    CK_THROW_(Error_t::WrongInput, "Overlapped pipeline is restricted to DLRM use");
  }
  if (!is_dlrm && grouped_all_reduce) {
    CK_THROW_(Error_t::WrongInput, "Grouped all reduce is restricted to DLRM use");
  }
  if (use_holistic_cuda_graph && use_cuda_graph) {
    CK_THROW_(Error_t::WrongInput, "Must turn off local cuda graph when using holistic cuda graph");
  }
  if (async_mlp_wgrad && use_cuda_graph) {
    CK_THROW_(Error_t::WrongInput, "Must turn off local cuda graph when using asynchronous wgrad computation of mlp");
  }

  std::unique_ptr<Solver> solver(new Solver());
  solver->seed = seed;
  solver->lr_policy = lr_policy;
  solver->lr = lr;
  solver->warmup_steps = warmup_steps;
  solver->decay_start = decay_start;
  solver->decay_steps = decay_steps;
  solver->decay_power = decay_power;
  solver->end_lr = end_lr;
  solver->max_eval_batches = max_eval_batches;
  solver->batchsize_eval = batchsize_eval;
  solver->batchsize = batchsize;
  solver->vvgpu.assign(vvgpu.begin(), vvgpu.end());
  solver->repeat_dataset = repeat_dataset;
  solver->use_mixed_precision = use_mixed_precision;
  solver->enable_tf32_compute = enable_tf32_compute;
  solver->scaler = scaler;
  solver->metrics_spec = metrics_spec;
  solver->i64_input_key = i64_input_key;
  solver->use_algorithm_search = use_algorithm_search;
  solver->use_cuda_graph = use_cuda_graph;
  solver->async_mlp_wgrad = async_mlp_wgrad;
  solver->gen_loss_summary = gen_loss_summary;
  solver->overlap_ar_a2a = overlap_ar_a2a;
  solver->device_layout = device_layout;
  solver->use_holistic_cuda_graph = use_holistic_cuda_graph;
  solver->use_overlapped_pipeline = use_overlapped_pipeline;
  solver->all_reduce_algo = all_reduce_algo;
  solver->grouped_all_reduce = grouped_all_reduce;
  solver->num_iterations_statistics = num_iterations_statistics;
  solver->is_dlrm = is_dlrm;
  return solver;
}

void SolverPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::Solver, std::unique_ptr<HugeCTR::Solver>>(m, "Solver")
      .def(pybind11::init<>())
      .def_readonly("seed", &HugeCTR::Solver::seed)
      .def_readonly("lr_policy", &HugeCTR::Solver::lr_policy)
      .def_readonly("lr", &HugeCTR::Solver::lr)
      .def_readonly("warmup_steps", &HugeCTR::Solver::warmup_steps)
      .def_readonly("decay_start", &HugeCTR::Solver::decay_start)
      .def_readonly("decay_steps", &HugeCTR::Solver::decay_steps)
      .def_readonly("decay_power", &HugeCTR::Solver::decay_power)
      .def_readonly("end_lr", &HugeCTR::Solver::end_lr)
      .def_readonly("max_eval_batches", &HugeCTR::Solver::max_eval_batches)
      .def_readonly("batchsize_eval", &HugeCTR::Solver::batchsize_eval)
      .def_readonly("batchsize", &HugeCTR::Solver::batchsize)
      .def_readonly("vvgpu", &HugeCTR::Solver::vvgpu)
      .def_readonly("repeat_dataset", &HugeCTR::Solver::repeat_dataset)
      .def_readonly("use_mixed_precision", &HugeCTR::Solver::use_mixed_precision)
      .def_readonly("enable_tf32_compute", &HugeCTR::Solver::enable_tf32_compute)
      .def_readonly("scaler", &HugeCTR::Solver::scaler)
      .def_readonly("metrics_spec", &HugeCTR::Solver::metrics_spec)
      .def_readonly("i64_input_key", &HugeCTR::Solver::i64_input_key)
      .def_readonly("use_algorithm_search", &HugeCTR::Solver::use_algorithm_search)
      .def_readonly("use_cuda_graph", &HugeCTR::Solver::use_cuda_graph)
      .def_readonly("async_mlp_wgrad", &HugeCTR::Solver::async_mlp_wgrad)
      .def_readonly("gen_loss_summary", &HugeCTR::Solver::gen_loss_summary)
      .def_readonly("overlap_ar_a2a", &HugeCTR::Solver::overlap_ar_a2a)
      .def_readonly("device_layout", &HugeCTR::Solver::device_layout)
      .def_readonly("use_holistic_cuda_graph", &HugeCTR::Solver::use_holistic_cuda_graph)
      .def_readonly("use_overlapped_pipeline", &HugeCTR::Solver::use_overlapped_pipeline)
      .def_readonly("all_reduce_algo", &HugeCTR::Solver::all_reduce_algo)
      .def_readonly("grouped_all_reduce", &HugeCTR::Solver::grouped_all_reduce)
      .def_readonly("num_iterations_statistics", &HugeCTR::Solver::num_iterations_statistics)
      .def_readonly("is_dlrm", &HugeCTR::Solver::is_dlrm);
  m.def("CreateSolver", &HugeCTR::python_lib::CreateSolver, pybind11::arg("seed") = 0,
        pybind11::arg("lr_policy") = LrPolicy_t::fixed, pybind11::arg("lr") = 0.001,
        pybind11::arg("warmup_steps") = 1, pybind11::arg("decay_start") = 0,
        pybind11::arg("decay_steps") = 1, pybind11::arg("decay_power") = 2.f,
        pybind11::arg("end_lr") = 0.f, pybind11::arg("max_eval_batches") = 100,
        pybind11::arg("batchsize_eval") = 2048, pybind11::arg("batchsize") = 2048,
        pybind11::arg("vvgpu") = std::vector<std::vector<int>>(1, std::vector<int>(1, 0)),
        pybind11::arg("repeat_dataset") = true, pybind11::arg("use_mixed_precision") = false,
        pybind11::arg("enable_tf32_compute") = false, pybind11::arg("scaler") = 1.f,
        pybind11::arg("metrics_spec") = std::map<metrics::Type, float>({{metrics::Type::AUC, 1.f}}),
        pybind11::arg("i64_input_key") = false, pybind11::arg("use_algorithm_search") = true,
        pybind11::arg("use_cuda_graph") = true,
        pybind11::arg("async_mlp_wgrad") = false,
        pybind11::arg("gen_loss_summary") = true,
        pybind11::arg("overlap_ar_a2a") = false,
        pybind11::arg("device_layout") = DeviceMap::Layout::LOCAL_FIRST,
        pybind11::arg("use_holistic_cuda_graph") = false,
        pybind11::arg("use_overlapped_pipeline") = false,
        pybind11::arg("all_reduce_algo") = AllReduceAlgo::NCCL,
        pybind11::arg("grouped_all_reduce") = false,
        pybind11::arg("num_iterations_statistics") = 20, pybind11::arg("is_dlrm") = false);
}

}  // namespace python_lib

}  // namespace HugeCTR

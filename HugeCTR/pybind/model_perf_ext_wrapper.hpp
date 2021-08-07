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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <HugeCTR/pybind/model_perf_ext.hpp>

namespace HugeCTR {

namespace python_lib {

void ModelPerfExtPybind(pybind11::module &m) {
  pybind11::class_<HugeCTR::ModelPerfExt, std::shared_ptr<HugeCTR::ModelPerfExt>>(m, "ModelPerfExt")
      .def(pybind11::init<const Solver &, const DataReaderParams &, std::shared_ptr<OptParamsPy> &,
                          std::shared_ptr<ModelOversubscriberParams> &>(),
           pybind11::arg("solver"), pybind11::arg("reader_params"), pybind11::arg("opt_params"),
           pybind11::arg("mos_params") =
               std::shared_ptr<ModelOversubscriberParams>(new ModelOversubscriberParams()))
      .def("compile", &HugeCTR::ModelPerfExt::compile)
      .def("summary", &HugeCTR::ModelPerfExt::summary)
      .def("graph_to_json", &HugeCTR::ModelPerfExt::graph_to_json, pybind11::arg("graph_config_file"))
      .def("fit", &HugeCTR::ModelPerfExt::fit, pybind11::arg("num_epochs") = 0,
           pybind11::arg("max_iter") = 2000, pybind11::arg("display") = 200,
           pybind11::arg("eval_interval") = 1000, pybind11::arg("snapshot") = 10000,
           pybind11::arg("snapshot_prefix") = "")
      .def("add", pybind11::overload_cast<Input &>(&HugeCTR::ModelPerfExt::add), pybind11::arg("input"))
      .def("add", pybind11::overload_cast<SparseEmbedding &>(&HugeCTR::ModelPerfExt::add),
           pybind11::arg("sparse_embedding"))
      .def("add", pybind11::overload_cast<DenseLayer &>(&HugeCTR::ModelPerfExt::add),
           pybind11::arg("dense_layer"));
}

}  // namespace python_lib

}  // namespace HugeCTR

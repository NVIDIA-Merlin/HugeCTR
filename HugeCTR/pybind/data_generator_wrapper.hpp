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

#include <data_generator.hpp>

namespace HugeCTR {

namespace python_lib {

void DataGeneratorPybind(pybind11::module &m) {
  pybind11::module tools = m.def_submodule("tools", "tools submodule of hugectr");
  pybind11::class_<HugeCTR::DataGeneratorParams, std::shared_ptr<HugeCTR::DataGeneratorParams>>(
      tools, "DataGeneratorParams")
      .def(pybind11::init<DataReaderType_t, int, int, int, bool, const std::string &,
                          const std::string &, const std::vector<size_t> &,
                          const std::vector<int> &, Check_t, Distribution_t, PowerLaw_t, float, int,
                          int, int, int, int, bool>(),
           pybind11::arg("format"), pybind11::arg("label_dim"), pybind11::arg("dense_dim"),
           pybind11::arg("num_slot"), pybind11::arg("i64_input_key"), pybind11::arg("source"),
           pybind11::arg("eval_source"), pybind11::arg("slot_size_array"),
           pybind11::arg("nnz_array") = std::vector<int>(),
           pybind11::arg("check_type") = Check_t::Sum,
           pybind11::arg("dist_type") = Distribution_t::PowerLaw,
           pybind11::arg("power_law_type") = PowerLaw_t::Specific, pybind11::arg("alpha") = 3.0,
           pybind11::arg("num_files") = 128, pybind11::arg("eval_num_files") = 32,
           pybind11::arg("num_samples_per_file") = 40960, pybind11::arg("num_samples") = 5242880,
           pybind11::arg("eval_num_samples") = 1310720, pybind11::arg("float_label_dense") = false)
      .def_readwrite("format", &HugeCTR::DataGeneratorParams::format)
      .def_readwrite("label_dim", &HugeCTR::DataGeneratorParams::label_dim)
      .def_readwrite("dense_dim", &HugeCTR::DataGeneratorParams::dense_dim)
      .def_readwrite("num_slot", &HugeCTR::DataGeneratorParams::num_slot)
      .def_readwrite("i64_input_key", &HugeCTR::DataGeneratorParams::i64_input_key)
      .def_readwrite("source", &HugeCTR::DataGeneratorParams::source)
      .def_readwrite("eval_source", &HugeCTR::DataGeneratorParams::eval_source)
      .def_readwrite("slot_size_array", &HugeCTR::DataGeneratorParams::slot_size_array)
      .def_readwrite("nnz_array", &HugeCTR::DataGeneratorParams::nnz_array)
      .def_readwrite("check_type", &HugeCTR::DataGeneratorParams::check_type)
      .def_readwrite("dist_type", &HugeCTR::DataGeneratorParams::dist_type)
      .def_readwrite("power_law_type", &HugeCTR::DataGeneratorParams::power_law_type)
      .def_readwrite("alpha", &HugeCTR::DataGeneratorParams::alpha)
      .def_readwrite("num_files", &HugeCTR::DataGeneratorParams::num_files)
      .def_readwrite("eval_num_files", &HugeCTR::DataGeneratorParams::eval_num_files)
      .def_readwrite("num_samples_per_file", &HugeCTR::DataGeneratorParams::num_samples_per_file)
      .def_readwrite("num_samples", &HugeCTR::DataGeneratorParams::num_samples)
      .def_readwrite("eval_num_samples", &HugeCTR::DataGeneratorParams::eval_num_samples)
      .def_readwrite("float_label_dense", &HugeCTR::DataGeneratorParams::float_label_dense);
  pybind11::class_<HugeCTR::DataGenerator, std::shared_ptr<HugeCTR::DataGenerator>>(tools,
                                                                                    "DataGenerator")
      .def(pybind11::init<const DataGeneratorParams &>(), pybind11::arg("data_generator_params"))
      .def("generate", &HugeCTR::DataGenerator::generate);
}

}  // namespace python_lib

}  // namespace HugeCTR
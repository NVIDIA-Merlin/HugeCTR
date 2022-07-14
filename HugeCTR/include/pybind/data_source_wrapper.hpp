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

#include <data_source/hdfs_backend.hpp>

namespace HugeCTR {

namespace python_lib {

void DataSourcePybind(pybind11::module &m) {
  pybind11::module data = m.def_submodule("data", "data submodule of hugectr");
  pybind11::class_<HugeCTR::DataSourceParams, std::shared_ptr<HugeCTR::DataSourceParams>>(
      data, "DataSourceParams")
      .def(pybind11::init<const bool, const std::string &, const int>(),
           pybind11::arg("use_hdfs") = false, pybind11::arg("namenode") = "localhost",
           pybind11::arg("port") = 9000)
      .def_readwrite("use_hdfs", &HugeCTR::DataSourceParams::use_hdfs)
      .def_readwrite("namenode", &HugeCTR::DataSourceParams::namenode)
      .def_readwrite("port", &HugeCTR::DataSourceParams::port);
  pybind11::class_<HugeCTR::DataSource, std::shared_ptr<HugeCTR::DataSource>>(data, "DataSource")
      .def(pybind11::init<const DataSourceParams &>(), pybind11::arg("data_source_params"))
      .def("move_to_local", &HugeCTR::DataSource::move_to_local);
}
}  // namespace python_lib
}  // namespace HugeCTR
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

#include <data_source/data_source_backend.hpp>

namespace HugeCTR {

namespace python_lib {

void DataSourcePybind(pybind11::module &m) {
  pybind11::module data = m.def_submodule("data", "data submodule of hugectr");
  pybind11::class_<HugeCTR::DataSourceParams, std::shared_ptr<HugeCTR::DataSourceParams>>(
      data, "DataSourceParams")
      .def(pybind11::init<DataSourceType_t, const std::string &, const int>(),
           pybind11::arg("source"), pybind11::arg("server"), pybind11::arg("port"))
      .def_readwrite("source", &HugeCTR::DataSourceParams::type)
      .def_readwrite("server", &HugeCTR::DataSourceParams::server)
      .def_readwrite("port", &HugeCTR::DataSourceParams::port);
}
}  // namespace python_lib
}  // namespace HugeCTR
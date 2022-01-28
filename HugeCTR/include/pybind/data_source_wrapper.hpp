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

#include <hdfs_backend.hpp>

namespace HugeCTR {

namespace python_lib {

void DataSourcePybind(pybind11::module &m) {
  pybind11::module data = m.def_submodule("data", "data submodule of hugectr");
  pybind11::class_<HugeCTR::DataSourceParams, std::shared_ptr<HugeCTR::DataSourceParams>>(
      data, "DataSourceParams")
      .def(pybind11::init<
               const bool, const std::string &, const int, const std::string &, const std::string &,
               const std::string &, const std::string &, const std::string &, const std::string &,
               const std::vector<std::string> &, const std::vector<std::string> &,
               const std::string &, const std::string &, const std::string &, const std::string &,
               const std::string &, const std::string &, const std::vector<std::string> &,
               const std::vector<std::string> &, const std::string &, const std::string &>(),
           pybind11::arg("use_hdfs") = false, pybind11::arg("namenode") = "localhost",
           pybind11::arg("port") = 9000, pybind11::arg("hdfs_train_source") = "",
           pybind11::arg("hdfs_train_filelist") = "", pybind11::arg("hdfs_eval_source") = "",
           pybind11::arg("hdfs_eval_filelist") = "", pybind11::arg("hdfs_dense_model") = "",
           pybind11::arg("hdfs_dense_opt_states") = "",
           pybind11::arg("hdfs_sparse_model") = std::vector<std::string>(),
           pybind11::arg("hdfs_sparse_opt_states") = std::vector<std::string>(),
           pybind11::arg("local_train_source") = "", pybind11::arg("local_train_filelist") = "",
           pybind11::arg("local_eval_source") = "", pybind11::arg("local_eval_filelist") = "",
           pybind11::arg("local_dense_model") = "", pybind11::arg("local_dense_opt_states") = "",
           pybind11::arg("local_sparse_model") = std::vector<std::string>(),
           pybind11::arg("local_sparse_opt_states") = std::vector<std::string>(),
           pybind11::arg("hdfs_model_home") = "", pybind11::arg("local_model_home") = "")
      .def_readwrite("use_hdfs", &HugeCTR::DataSourceParams::use_hdfs)
      .def_readwrite("namenode", &HugeCTR::DataSourceParams::namenode)
      .def_readwrite("port", &HugeCTR::DataSourceParams::port)
      .def_readwrite("hdfs_train_source", &HugeCTR::DataSourceParams::hdfs_train_source)
      .def_readwrite("hdfs_train_filelist", &HugeCTR::DataSourceParams::hdfs_train_filelist)
      .def_readwrite("hdfs_eval_source", &HugeCTR::DataSourceParams::hdfs_eval_source)
      .def_readwrite("hdfs_eval_filelist", &HugeCTR::DataSourceParams::hdfs_eval_filelist)
      .def_readwrite("hdfs_dense_model", &HugeCTR::DataSourceParams::hdfs_dense_model)
      .def_readwrite("hdfs_dense_opt_states", &HugeCTR::DataSourceParams::hdfs_dense_opt_states)
      .def_readwrite("hdfs_sparse_model", &HugeCTR::DataSourceParams::hdfs_sparse_model)
      .def_readwrite("hdfs_sparse_opt_states", &HugeCTR::DataSourceParams::hdfs_sparse_opt_states)
      .def_readwrite("local_train_source", &HugeCTR::DataSourceParams::local_train_source)
      .def_readwrite("local_train_filelist", &HugeCTR::DataSourceParams::local_train_filelist)
      .def_readwrite("local_eval_source", &HugeCTR::DataSourceParams::local_eval_source)
      .def_readwrite("local_eval_filelist", &HugeCTR::DataSourceParams::local_eval_filelist)
      .def_readwrite("local_dense_model", &HugeCTR::DataSourceParams::local_dense_model)
      .def_readwrite("local_dense_opt_states", &HugeCTR::DataSourceParams::local_dense_opt_states)
      .def_readwrite("local_sparse_model", &HugeCTR::DataSourceParams::local_sparse_model)
      .def_readwrite("local_sparse_opt_states", &HugeCTR::DataSourceParams::local_sparse_opt_states)
      .def_readwrite("hdfs_model_home", &HugeCTR::DataSourceParams::hdfs_model_home)
      .def_readwrite("local_model_home", &HugeCTR::DataSourceParams::local_model_home);
  pybind11::class_<HugeCTR::DataSource, std::shared_ptr<HugeCTR::DataSource>>(data, "DataSource")
      .def(pybind11::init<const DataSourceParams &>(), pybind11::arg("data_source_params"))
      .def("move_to_local", &HugeCTR::DataSource::move_to_local);
}
}  // namespace python_lib
}  // namespace HugeCTR
/*
 * @Author: your name
 * @Date: 2020-11-06 16:26:09
 * @LastEditTime: 2020-11-06 16:26:23
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /hugectr/HugeCTR/pybind/model_oversubscriber_wrapper.hpp
 */
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
#include <HugeCTR/include/model_oversubscriber/model_oversubscriber.hpp>

namespace HugeCTR {

namespace python_lib {

void ModelOversubscriberPybind(pybind11::module &m) {
  pybind11::class_<HugeCTR::ModelOversubscriber, std::shared_ptr<HugeCTR::ModelOversubscriber>>(m, "ModelOversubscriber")
    .def("store", &HugeCTR::ModelOversubscriber::store,
	 pybind11::arg("snapshot_file_list"))
    .def("update", pybind11::overload_cast<std::string&>(&HugeCTR::ModelOversubscriber::update),
         pybind11::arg("keyset_file"))
    .def("update", pybind11::overload_cast<std::vector<std::string>&>(&HugeCTR::ModelOversubscriber::update),
         pybind11::arg("keyset_file_list"));
}

}  //  namespace python_lib

}  //  namespace HugeCTR


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
//#include <mpi.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rmm/device_buffer.hpp>
#include <thread>
namespace HugeCTR {

namespace python_lib {

void ThirdPartyPybind(pybind11::module& m) {
  pybind11::class_<rmm::mr::device_memory_resource,
                   std::shared_ptr<rmm::mr::device_memory_resource>>(m, "device_memory_resource");
  pybind11::class_<rmm::mr::cuda_memory_resource, std::shared_ptr<rmm::mr::cuda_memory_resource>,
                   rmm::mr::device_memory_resource>(m, "cuda_memory_resource")
      .def(pybind11::init<>());
  pybind11::class_<std::thread>(m, "thread")
      .def(pybind11::init<std::function<void()>>(), pybind11::arg("target"))
      .def("join", &std::thread::join);
}

}  // namespace python_lib

}  // namespace HugeCTR

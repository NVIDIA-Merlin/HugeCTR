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
#include <HugeCTR/include/device_map.hpp>

namespace HugeCTR {

namespace python_lib {

void DeviceMapPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::DeviceMap, std::shared_ptr<DeviceMap>>(m, "DeviceMap")
      .def(pybind11::init<const std::vector<std::vector<int>>, int>(),
           pybind11::arg("device_list_total"), pybind11::arg("my_pid"));
}

}  // namespace python_lib

}  // namespace HugeCTR

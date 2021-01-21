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
#include <HugeCTR/include/data_readers/mmap_offset_list.hpp>

namespace HugeCTR {

namespace python_lib {

void MmapOffsetPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::MmapOffsetList, std::shared_ptr<HugeCTR::MmapOffsetList>>(
      m, "MmapOffsetList")
      .def(pybind11::init<std::string, long long, long long, long long, bool, int, bool>(),
           pybind11::arg("file_name"), pybind11::arg("num_samples"), pybind11::arg("stride"),
           pybind11::arg("batchsize"), pybind11::arg("use_shuffle"), pybind11::arg("num_workers"),
           pybind11::arg("repeat"))
      .def("get_offset", &HugeCTR::MmapOffsetList::get_offset, pybind11::arg("round"),
           pybind11::arg("worker_id"));
}

}  // namespace python_lib

}  // namespace HugeCTR

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

void LearningRateSchedulerPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::LearningRateScheduler, std::shared_ptr<HugeCTR::LearningRateScheduler>>(m, "LearningRateScheduler")
    .def("get_next", &HugeCTR::LearningRateScheduler::get_next);
}

}  // namespace python_lib

}  // namespace HugeCTR

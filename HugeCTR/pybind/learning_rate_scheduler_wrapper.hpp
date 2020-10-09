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
//#include <HugeCTR/include/learning_rate_scheduler.hpp>
#include <HugeCTR/include/parser.hpp>

namespace HugeCTR {

namespace python_lib {

void LearningRateSchedulerPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::LearningRateScheduler, std::unique_ptr<HugeCTR::LearningRateScheduler>>(
      m, "LearningRateScheduler")
      .def(pybind11::init<float, size_t, size_t, size_t, float, float>(), pybind11::arg("base_lr"),
           pybind11::arg("warmup_steps") = 1, pybind11::arg("decay_start") = 0,
           pybind11::arg("decay_steps") = 1, pybind11::arg("decay_power") = 2.f,
           pybind11::arg("end_lr") = 0.f)
      .def("get_next", &HugeCTR::LearningRateScheduler::get_next)
      .def("get_lr", &HugeCTR::LearningRateScheduler::get_lr)
      .def("get_step", &HugeCTR::LearningRateScheduler::get_step);
  m.def("get_learning_rate_scheduler", &HugeCTR::get_learning_rate_scheduler,
        pybind11::arg("configure_file"));
}

}  // namespace python_lib

}  // namespace HugeCTR

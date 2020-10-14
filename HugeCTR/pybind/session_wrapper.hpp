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
#include <HugeCTR/include/session.hpp>

namespace HugeCTR {

namespace python_lib {

void SessionPybind(pybind11::module &m) {
  pybind11::class_<HugeCTR::Session, std::shared_ptr<HugeCTR::Session>>(m, "Session")
      .def(pybind11::init<const SolverParser&, const std::string&>(),
		  pybind11::arg("solver_config"),
                  pybind11::arg("config_file"))
      .def("train", &HugeCTR::Session::train)
      .def("eval", &HugeCTR::Session::eval)
      .def("get_eval_metrics", &HugeCTR::Session::get_eval_metrics)
      .def("evaluation",
           [](HugeCTR::Session &self) {
             self.eval();
             std::vector<std::pair<std::string, float>> metrics = self.get_eval_metrics();
             return metrics;
           })
      .def("start_data_reading", &HugeCTR::Session::start_data_reading)
      .def("get_current_loss",
           [](HugeCTR::Session &self) {
             float loss = 0;
             self.get_current_loss(&loss);
             return loss;
           })
      .def("download_params_to_files", &HugeCTR::Session::download_params_to_files,
           pybind11::arg("prefix"), pybind11::arg("iter"))
      .def("set_learning_rate", &HugeCTR::Session::set_learning_rate,
           pybind11::arg("lr"))
      .def("init_params", &HugeCTR::Session::init_params,
           pybind11::arg("model_file"))
      .def("get_params_num", &HugeCTR::Session::get_params_num)
      .def("check_overflow", &HugeCTR::Session::check_overflow);
}

}  //  namespace python_lib

}  //  namespace HugeCTR

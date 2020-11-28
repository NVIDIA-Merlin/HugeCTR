/*
 * @Author: your name
 * @Date: 2020-11-06 15:00:46
 * @LastEditTime: 2020-11-06 16:25:36
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /hugectr/HugeCTR/pybind/session_wrapper.hpp
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
#include <HugeCTR/include/session.hpp>

namespace HugeCTR {

namespace python_lib {

void SessionPybind(pybind11::module &m) {
  pybind11::class_<HugeCTR::Session, std::shared_ptr<HugeCTR::Session>>(m, "Session")
      .def(pybind11::init<const SolverParser&, const std::string&,
		          bool, const std::string>(),
		  pybind11::arg("solver_config"),
                  pybind11::arg("config_file"),
		  pybind11::arg("use_model_oversubscriber")=false,
		  pybind11::arg("temp_embedding_dir")=std::string())
      .def("train", &HugeCTR::Session::train)
      .def("eval", &HugeCTR::Session::eval)
      .def("get_eval_metrics", &HugeCTR::Session::get_eval_metrics)
      .def("evaluation",
           [](HugeCTR::Session &self) {
             bool good = false;
	     do {
	       good = self.eval();
	       if (!good) {
	         auto data_reader_eval = self.get_data_reader_eval();
		 data_reader_eval->set_file_list_source();
	       }
	     } while(!good);
	     auto metrics = self.get_eval_metrics();
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
      .def("get_model_oversubscriber", &HugeCTR::Session::get_model_oversubscriber)
      .def("init_params", &HugeCTR::Session::init_params,
           pybind11::arg("model_file"))
      .def("get_params_num", &HugeCTR::Session::get_params_num)
      .def("check_overflow", &HugeCTR::Session::check_overflow)
      .def("get_data_reader_train", &HugeCTR::Session::get_data_reader_train)
      .def("get_data_reader_eval", &HugeCTR::Session::get_data_reader_eval);
}

}  //  namespace python_lib

}  //  namespace HugeCTR

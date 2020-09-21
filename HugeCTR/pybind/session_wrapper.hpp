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
  
void SessionPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::Session, std::shared_ptr<HugeCTR::Session>>(m, "Session")
    .def("Create", &HugeCTR::Session::Create,
		    pybind11::arg("solver_config"));
  pybind11::class_<HugeCTR::SessionImpl<long long>, std::shared_ptr<HugeCTR::SessionImpl<long long>>, HugeCTR::Session>(m, "SessionImpl64")
      //.def(pybind11::init<>())
    .def("train", &HugeCTR::SessionImpl<long long>::train)
    .def("eval", &HugeCTR::SessionImpl<long long>::eval)
    .def("get_eval_metrics", &HugeCTR::SessionImpl<long long>::get_eval_metrics)
    .def("start_data_reading", &HugeCTR::SessionImpl<long long>::start_data_reading)
    .def("get_current_loss", [](HugeCTR::SessionImpl<long long> &self) {
		                    float loss = 0;
				    self.get_current_loss(&loss);
				    return loss;
				  })
    .def("download_params_to_files", &HugeCTR::SessionImpl<long long>::download_params_to_files,
		                      pybind11::arg("prefix"),
				      pybind11::arg("iter"))
    .def("set_learning_rate", &HugeCTR::SessionImpl<long long>::set_learning_rate,
		               pybind11::arg("lr"))
    .def("init_params", &HugeCTR::SessionImpl<long long>::init_params,
		         pybind11::arg("model_file"))
    .def("get_params_num", &HugeCTR::SessionImpl<long long>::get_params_num)
    .def("check_overflow", &HugeCTR::SessionImpl<long long>::check_overflow);
    
  pybind11::class_<HugeCTR::SessionImpl<unsigned int>, std::shared_ptr<HugeCTR::SessionImpl<unsigned int>>, HugeCTR::Session>(m, "SessionImpl32")
    .def("train", &HugeCTR::SessionImpl<unsigned int>::train)
    .def("eval", &HugeCTR::SessionImpl<unsigned int>::eval)
    .def("get_eval_metrics", &HugeCTR::SessionImpl<unsigned int>::get_eval_metrics)
    .def("start_data_reading", &HugeCTR::SessionImpl<unsigned int>::start_data_reading)
    .def("get_current_loss", [](HugeCTR::SessionImpl<unsigned int> &self) {
                                    float loss = 0;
                                    self.get_current_loss(&loss);
                                    return loss;
                                  })
    .def("download_params_to_files", &HugeCTR::SessionImpl<unsigned int>::download_params_to_files,
		                      pybind11::arg("prefix"),
				      pybind11::arg("iter"))
    .def("set_learning_rate", &HugeCTR::SessionImpl<unsigned int>::set_learning_rate,
		               pybind11::arg("lr"))
    .def("init_params", &HugeCTR::SessionImpl<unsigned int>::init_params,
		         pybind11::arg("model_file"))
    .def("get_params_num", &HugeCTR::SessionImpl<unsigned int>::get_params_num)
    .def("check_overflow", &HugeCTR::SessionImpl<unsigned int>::check_overflow);
}

}  //  namespace python_lib 

}  //  namespace HugeCTR




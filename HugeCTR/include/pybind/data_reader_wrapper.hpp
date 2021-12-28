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

#include <HugeCTR/include/data_readers/data_reader.hpp>

namespace HugeCTR {

namespace python_lib {

void DataReaderPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::IDataReader, std::shared_ptr<HugeCTR::IDataReader>>(m, "IDataReader");
  pybind11::class_<HugeCTR::DataReader<long long>, std::shared_ptr<HugeCTR::DataReader<long long>>,
                   HugeCTR::IDataReader>(m, "DataReader64")
      .def("set_source", &HugeCTR::DataReader<long long>::set_source,
           pybind11::arg("file_name") = std::string())
      .def("is_started", &HugeCTR::DataReader<long long>::is_started)
      .def("ready_to_collect", &HugeCTR::DataReader<long long>::ready_to_collect)
      .def("read_a_batch_to_device", &HugeCTR::DataReader<long long>::read_a_batch_to_device)
      .def("read_a_batch_to_device_delay_release",
           &HugeCTR::DataReader<long long>::read_a_batch_to_device_delay_release);
  pybind11::class_<HugeCTR::DataReader<unsigned int>,
                   std::shared_ptr<HugeCTR::DataReader<unsigned int>>, HugeCTR::IDataReader>(
      m, "DataReader32")
      .def("set_source", &HugeCTR::DataReader<unsigned int>::set_source,
           pybind11::arg("file_name") = std::string())
      .def("is_started", &HugeCTR::DataReader<unsigned int>::is_started)
      .def("ready_to_collect", &HugeCTR::DataReader<unsigned int>::ready_to_collect)
      .def("read_a_batch_to_device", &HugeCTR::DataReader<unsigned int>::read_a_batch_to_device)
      .def("read_a_batch_to_device_delay_release",
           &HugeCTR::DataReader<unsigned int>::read_a_batch_to_device_delay_release);
}

}  // namespace python_lib

}  // namespace HugeCTR

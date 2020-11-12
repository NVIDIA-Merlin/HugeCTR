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
#include <HugeCTR/include/data_readers/csr_chunk.hpp>

namespace HugeCTR {

namespace python_lib {

void CSRChunkPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::CSRChunk<long long>>(m, "CSRChunk")
      .def(pybind11::init<int, int, int, const std::vector<DataReaderSparseParam>&>(),
           pybind11::arg("num_devices"), pybind11::arg("batchsize"),
           pybind11::arg("label_dense_dim"), pybind11::arg("params"))
      .def("set_current_batchsize", &HugeCTR::CSRChunk<long long>::set_current_batchsize,
           pybind11::arg("current_batchsize"))
      .def("get_current_batchsize", &HugeCTR::CSRChunk<long long>::get_current_batchsize)
      .def("get_csr_buffers", &HugeCTR::CSRChunk<long long>::get_csr_buffers)
      .def("get_csr_buffer",
           pybind11::overload_cast<int>(&HugeCTR::CSRChunk<long long>::get_csr_buffer),
           pybind11::arg("i"))
      .def("get_csr_buffer",
           pybind11::overload_cast<int, int>(&HugeCTR::CSRChunk<long long>::get_csr_buffer),
           pybind11::arg("param_id"), pybind11::arg("dev_id"))
      .def("get_label_buffers", &HugeCTR::CSRChunk<long long>::get_label_buffers)
      .def("get_label_dense_dim", &HugeCTR::CSRChunk<long long>::get_label_dense_dim)
      .def("get_batchsize", &HugeCTR::CSRChunk<long long>::get_batchsize)
      .def("get_num_devices", &HugeCTR::CSRChunk<long long>::get_num_devices)
      .def("get_num_params", &HugeCTR::CSRChunk<long long>::get_num_params);
}

}  // namespace python_lib

}  // namespace HugeCTR

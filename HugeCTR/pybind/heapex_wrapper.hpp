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
#include <HugeCTR/include/data_readers/heapex.hpp>

namespace HugeCTR {

namespace python_lib {

void HeapExPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::HeapEx<CSRChunk<long long>>,
                   std::shared_ptr<HugeCTR::HeapEx<CSRChunk<long long>>>>(m, "HeapEx")
      .def(pybind11::init<int, int, int, long long, const std::vector<DataReaderSparseParam>&>(),
           pybind11::arg("num"), pybind11::arg("num_devices"), pybind11::arg("batchsize"),
           pybind11::arg("dim"), pybind11::arg("params"))
      //.def("free_chunk_checkout", &HugeCTR::HeapEx<CSRChunk<long long>>::free_chunk_checkout)
      .def("chunk_write_and_checkin",
           &HugeCTR::HeapEx<CSRChunk<long long>>::chunk_write_and_checkin, pybind11::arg("id"))
      //.def("data_chunk_checkout", &HugeCTR::HeapEx<CSRChunk<long long>>::data_chunk_checkout)
      .def("chunk_free_and_checkin", &HugeCTR::HeapEx<CSRChunk<long long>>::chunk_free_and_checkin)
      .def("break_and_return", &HugeCTR::HeapEx<CSRChunk<long long>>::break_and_return);
}

}  // namespace python_lib

}  // namespace HugeCTR

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
#include <HugeCTR/include/data_readers/data_reader.hpp>
#include <HugeCTR/include/data_readers/data_reader_worker.hpp>
#include <HugeCTR/include/data_readers/data_reader_worker_interface.hpp>
#include <HugeCTR/include/data_readers/data_reader_worker_raw.hpp>
#include <HugeCTR/include/data_readers/parquet_data_reader_worker.hpp>

namespace HugeCTR {

namespace python_lib {

void DataReaderPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::IDataReaderWorker>(m, "IDataReaderWorker");
  pybind11::class_<HugeCTR::ParquetDataReaderWorker<long long>, HugeCTR::IDataReaderWorker>(
      m, "ParquetDataReaderWorker64")
      .def(pybind11::init<unsigned int, unsigned int,
                          const std::shared_ptr<HeapEx<CSRChunk<long long>>>&, const std::string&,
                          size_t, const std::vector<DataReaderSparseParam>&,
                          const std::vector<long long>&,
                          const std::shared_ptr<rmm::mr::device_memory_resource>&>(),
           pybind11::arg("worker_id"), pybind11::arg("worker_num"), pybind11::arg("csr_heap"),
           pybind11::arg("file_list"), pybind11::arg("buffer_length"), pybind11::arg("params"),
           pybind11::arg("slot_offset"), pybind11::arg("mr"))
      .def("read_a_batch", &HugeCTR::ParquetDataReaderWorker<long long>::read_a_batch)
      .def("skip_read", &HugeCTR::ParquetDataReaderWorker<long long>::skip_read);
  pybind11::class_<HugeCTR::DataReaderWorkerRaw<long long>, HugeCTR::IDataReaderWorker>(
      m, "DataReaderWorkerRaw64")
      .def(pybind11::init<unsigned int, unsigned int, std::shared_ptr<MmapOffsetList>&,
                          const std::shared_ptr<HeapEx<CSRChunk<long long>>>&, const std::string,
                          const std::vector<DataReaderSparseParam>&, const std::vector<long long>&,
                          int, bool>(),
           pybind11::arg("worker_id"), pybind11::arg("worker_num"),
           pybind11::arg("file_offset_list"), pybind11::arg("csr_heap"), pybind11::arg("file_name"),
           pybind11::arg("params"), pybind11::arg("slot_offset"), pybind11::arg("label_dim"),
           pybind11::arg("float_label_dense"))
      .def("read_a_batch", &HugeCTR::DataReaderWorkerRaw<long long>::read_a_batch)
      .def("skip_read", &HugeCTR::DataReaderWorkerRaw<long long>::skip_read);
  pybind11::class_<HugeCTR::DataReaderWorker<long long>, HugeCTR::IDataReaderWorker>(
      m, "DataReaderWorker64")
      .def(pybind11::init<unsigned int, unsigned int,
                          const std::shared_ptr<HeapEx<CSRChunk<long long>>>&, const std::string&,
                          size_t, Check_t, const std::vector<DataReaderSparseParam>&>(),
           pybind11::arg("worker_id"), pybind11::arg("worker_num"), pybind11::arg("csr_heap"),
           pybind11::arg("file_list"), pybind11::arg("buffer_length"), pybind11::arg("check_type"),
           pybind11::arg("params"))
      .def("read_a_batch", &HugeCTR::DataReaderWorker<long long>::read_a_batch)
      .def("skip_read", &HugeCTR::DataReaderWorker<long long>::skip_read);
  
  pybind11::class_<HugeCTR::IDataReader, std::unique_ptr<HugeCTR::IDataReader>>(m, "IDataReader");
  pybind11::class_<HugeCTR::DataReader<long long>, std::unique_ptr<HugeCTR::DataReader<long long>>, HugeCTR::IDataReader>(m, "DataReader64")
      .def(pybind11::init<int, size_t, int, std::vector<DataReaderSparseParam>&,
                          const std::shared_ptr<ResourceManager>&, int, bool, int>(),
           pybind11::arg("batchsize"), pybind11::arg("label_dim"), pybind11::arg("dense_dim"),
           pybind11::arg("params"), pybind11::arg("resource_manager"),
           pybind11::arg("num_chunk_threads") = 31, pybind11::arg("use_mixed_precision") = false,
           pybind11::arg("cache_num_iters") = 0)
      .def("create_drwg_norm", &HugeCTR::DataReader<long long>::create_drwg_norm,
           pybind11::arg("file_list"), pybind11::arg("Check_t"),
           pybind11::arg("start_reading_from_beginning") = true)
      .def("create_drwg_raw", &HugeCTR::DataReader<long long>::create_drwg_raw,
           pybind11::arg("file_name"), pybind11::arg("num_samples"), pybind11::arg("slot_offset"),
           pybind11::arg("float_label_dense"), pybind11::arg("data_shuffle") = false,
           pybind11::arg("start_reading_from_beginning") = true)
      .def("create_drwg_parquet", &HugeCTR::DataReader<long long>::create_drwg_parquet,
           pybind11::arg("file_list"), pybind11::arg("slot_offset"),
           pybind11::arg("start_reading_from_beginning") = true)
      .def("read_a_batch_to_device", &HugeCTR::DataReader<long long>::read_a_batch_to_device)
      .def("read_a_batch_to_device_delay_release",
           &HugeCTR::DataReader<long long>::read_a_batch_to_device_delay_release)
      .def("ready_to_collect", &HugeCTR::DataReader<long long>::ready_to_collect)
      .def("start", &HugeCTR::DataReader<long long>::start)
      .def("get_label_tensors", &HugeCTR::DataReader<long long>::get_label_tensors)
      .def("get_dense_tensors", &HugeCTR::DataReader<long long>::get_dense_tensors)
      .def("get_row_offsets_tensors",
           pybind11::overload_cast<>(&HugeCTR::DataReader<long long>::get_row_offsets_tensors,
                                     pybind11::const_))
      .def("get_value_tensors",
           pybind11::overload_cast<>(&HugeCTR::DataReader<long long>::get_value_tensors,
                                     pybind11::const_))
      .def("get_row_offsets_tensors",
           pybind11::overload_cast<int>(&HugeCTR::DataReader<long long>::get_row_offsets_tensors,
                                        pybind11::const_),
           pybind11::arg("param_id"))
      .def("get_value_tensors",
           pybind11::overload_cast<int>(&HugeCTR::DataReader<long long>::get_value_tensors,
                                        pybind11::const_),
           pybind11::arg("param_id"));

  pybind11::class_<HugeCTR::DataReader<unsigned int>, std::unique_ptr<HugeCTR::DataReader<unsigned int>>, HugeCTR::IDataReader>(m, "DataReader32")
      .def(pybind11::init<int, size_t, int, std::vector<DataReaderSparseParam>&,
                          const std::shared_ptr<ResourceManager>&, int, bool, int>(),
           pybind11::arg("batchsize"), pybind11::arg("label_dim"), pybind11::arg("dense_dim"),
           pybind11::arg("params"), pybind11::arg("resource_manager"),
           pybind11::arg("num_chunk_threads") = 31, pybind11::arg("use_mixed_precision") = false,
           pybind11::arg("cache_num_iters") = 0)
      .def("create_drwg_norm", &HugeCTR::DataReader<unsigned int>::create_drwg_norm,
           pybind11::arg("file_list"), pybind11::arg("Check_t"),
           pybind11::arg("start_reading_from_beginning") = true)
      .def("create_drwg_raw", &HugeCTR::DataReader<unsigned int>::create_drwg_raw,
           pybind11::arg("file_name"), pybind11::arg("num_samples"), pybind11::arg("slot_offset"),
           pybind11::arg("float_label_dense"), pybind11::arg("data_shuffle") = false,
           pybind11::arg("start_reading_from_beginning") = true)
      .def("create_drwg_parquet", &HugeCTR::DataReader<unsigned int>::create_drwg_parquet,
           pybind11::arg("file_list"), pybind11::arg("slot_offset"),
           pybind11::arg("start_reading_from_beginning") = true)
      .def("read_a_batch_to_device", &HugeCTR::DataReader<unsigned int>::read_a_batch_to_device)
      .def("read_a_batch_to_device_delay_release",
           &HugeCTR::DataReader<unsigned int>::read_a_batch_to_device_delay_release)
      .def("ready_to_collect", &HugeCTR::DataReader<unsigned int>::ready_to_collect)
      .def("start", &HugeCTR::DataReader<unsigned int>::start)
      .def("get_label_tensors", &HugeCTR::DataReader<unsigned int>::get_label_tensors)
      .def("get_dense_tensors", &HugeCTR::DataReader<unsigned int>::get_dense_tensors)
      .def("get_row_offsets_tensors",
           pybind11::overload_cast<>(&HugeCTR::DataReader<unsigned int>::get_row_offsets_tensors,
                                     pybind11::const_))
      .def("get_value_tensors",
           pybind11::overload_cast<>(&HugeCTR::DataReader<unsigned int>::get_value_tensors,
                                     pybind11::const_))
      .def("get_row_offsets_tensors",
           pybind11::overload_cast<int>(&HugeCTR::DataReader<unsigned int>::get_row_offsets_tensors,
                                        pybind11::const_),
           pybind11::arg("param_id"))
      .def("get_value_tensors",
           pybind11::overload_cast<int>(&HugeCTR::DataReader<unsigned int>::get_value_tensors,
                                        pybind11::const_),
           pybind11::arg("param_id"));
}

}  // namespace python_lib

}  // namespace HugeCTR

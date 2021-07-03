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

#include "HugeCTR/include/data_readers/data_reader.hpp"
#include <fstream>
#include "HugeCTR/include/data_readers/parquet_data_reader_worker.hpp"
#include "HugeCTR/include/data_readers/file_list.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"
#include <HugeCTR/include/resource_managers/resource_manager_ext.hpp>
#include "HugeCTR/include/data_generator.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/io/parquet.hpp>
#include <rmm/device_buffer.hpp>
#pragma GCC diagnostic pop

using namespace HugeCTR;

const std::vector<long long> slot_size = {
    39884406, 39043,    17289,    7420,     20263,  3,     7120, 1543, 63,
    38532951, 2953546,  403346,   10,       2208,   11938, 155,  4,    976,
    14,       39979771, 25641295, 39664984, 585935, 12972, 108,  36};

const int max_nnz = 1;
const int slot_num = 26;
const int label_dim = 1;
const int dense_dim = 13;
using LABEL_TYPE = float;
using DENSE_TYPE = float;
using CAT_TYPE = int64_t;

const Check_t CHK = Check_t::None;
const std::string prefix("./data_reader_parquet_test_data/");
const std::string file_list_name("data_reader_parquet_file_list.txt");
using CVector     = std::vector<std::unique_ptr<cudf::column>>;
typedef long long T;

void generate_parquet_input_files(int num_files, int sample_per_file) {
  check_make_dir(prefix);
  for (int file_num = 0; file_num < num_files; file_num++) {
    CVector cols;
    // create label vector
    for (int i = 0; i < label_dim; i++) {
      std::vector<LABEL_TYPE> label_vector(sample_per_file, 0);
      for (auto& c : label_vector)
        c = LABEL_TYPE(std::rand() % 2);  // 0,1 value

      rmm::device_buffer dev_buffer(label_vector.data(), sizeof(LABEL_TYPE) * sample_per_file);
      cols.emplace_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<LABEL_TYPE>()},
                                  cudf::size_type(sample_per_file),
                                  dev_buffer));
    }

    // create dense vector
    for (int i = 0; i < dense_dim; i++) {
      std::vector<DENSE_TYPE> dense_vector(sample_per_file, 0);
      for (auto& c : dense_vector)
        c = DENSE_TYPE(std::rand());

      rmm::device_buffer dev_buffer(dense_vector.data(), sizeof(DENSE_TYPE) * sample_per_file);
      cols.emplace_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<DENSE_TYPE>()},
                                  cudf::size_type(sample_per_file),
                                  dev_buffer));
    }

    // create slot vectors
    for (int i = 0; i < slot_num; i++) {
      std::vector<CAT_TYPE> slot_vector(sample_per_file, 0);
      for (auto& c : slot_vector)
        c = CAT_TYPE(std::rand());

      rmm::device_buffer dev_buffer(slot_vector.data(), sizeof(CAT_TYPE) * sample_per_file);
      cols.emplace_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<CAT_TYPE>()},
                                  cudf::size_type(sample_per_file),
                                  dev_buffer));
    }

    cudf::table input_table(std::move(cols));

    std::string filepath = prefix + std::to_string(file_num) + std::string(".parquet");
    cudf::io::parquet_writer_options writer_args = cudf_io::parquet_writer_options::builder(
                                                        cudf::io::sink_info{filepath},
                                                        input_table.view());
    cudf::io::write_parquet(writer_args);
  }

  std::ofstream output_file_stream;
  output_file_stream.open(file_list_name, std::ofstream::out);
  output_file_stream << num_files << std::endl;
  for (int i = 0; i < num_files; i++) {
    std::string filepath = prefix + "./" + std::to_string(i) + std::string(".parquet");
    output_file_stream <<  filepath << std::endl;
  }
  output_file_stream.close();


  // also write metadata
  std::stringstream metadata;
  metadata << "{ \"file_stats\": [";
  for (int i = 0; i < num_files-1; i++) {
    std::string filepath = std::to_string(i) + std::string(".parquet");
    metadata << "{\"file_name\": \"" << filepath << "\", " << "\"num_rows\":" << sample_per_file << "}, ";
  }
  std::string filepath = std::to_string(num_files-1) + std::string(".parquet");
  metadata << "{\"file_name\": \"" << filepath << "\", " << "\"num_rows\":" << sample_per_file << "} ";
  metadata << "], ";
  metadata << "\"labels\": [";
  for (int i = 0; i < label_dim; i++) {
    metadata << "{\"col_name\": \"label\", " << "\"index\":" << i << "} ";
  }
  metadata << "], ";
  
  metadata << "\"conts\": [";
  for (int i = label_dim; i < (label_dim+dense_dim - 1); i++) {
    metadata << "{\"col_name\": \"C" << i << "\", " << "\"index\":" << i << "}, ";
  }
  metadata << "{\"col_name\": \"C" << (label_dim+dense_dim - 1) << "\", " << "\"index\":" \
                  << (label_dim+dense_dim - 1) << "} ";
  metadata << "], ";

  metadata << "\"cats\": [";
  for (int i = label_dim+dense_dim; i < (label_dim+dense_dim+slot_num - 1); i++) {
    metadata << "{\"col_name\": \"C" << i << "\", " << "\"index\":" << i << "}, ";
  }
  metadata << "{\"col_name\": \"C" << (label_dim+dense_dim+slot_num - 1) << "\", " \
                  << "\"index\":" << (label_dim+dense_dim+slot_num - 1) << "} ";
  metadata << "] ";
  metadata << "}";

  std::ofstream metadata_file_stream;
  metadata_file_stream.open(prefix + "_metadata.json", std::ofstream::out);
  metadata_file_stream << metadata.rdbuf();
  metadata_file_stream.close();
}

TEST(data_reader_parquet_worker, data_reader_parquet_worker_distributed_test) {
  auto p_mr = rmm::mr::get_current_device_resource();
  generate_parquet_input_files(3, 1024);

  // setup a CSR heap
  const int num_devices = 1;
  const int batchsize = 128;

  int numprocs = 1;
  std::vector<std::vector<int>> vvgpu;
  std::vector<int> device_list = {0};
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  auto gpu_resource_group = ResourceManagerExt::create(vvgpu, 0);
  
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, max_nnz * slot_num, max_nnz,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  constexpr size_t buffer_length = max_nnz;
  std::shared_ptr<HeapEx<CSRChunk<T>>> csr_heap(
      new HeapEx<CSRChunk<T>>(1, num_devices, batchsize, label_dim + dense_dim, params));

  std::vector<long long> slot_offset(slot_size.size(), 0);
  for (unsigned int i = 1; i < slot_size.size(); i++) {
    slot_offset[i] = slot_offset[i-1] + slot_size[i-1];
  }

  // setup a data reader
  ParquetDataReaderWorker<T> data_reader(0, 1, csr_heap, file_list_name, buffer_length,
                                          params, slot_offset, 0, gpu_resource_group);

  // call read a batch
  data_reader.read_a_batch();

  rmm::mr::set_current_device_resource(p_mr);
}

TEST(data_reader_parquet_worker, data_reader_parquet_worker_localized_test) {
  auto p_mr = rmm::mr::get_current_device_resource();
  generate_parquet_input_files(3, 2048);

  int numprocs = 1;
  std::vector<std::vector<int>> vvgpu;
  std::vector<int> device_list = {0};
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  auto gpu_resource_group = ResourceManagerExt::create(vvgpu, 0);
  // setup a CSR heap
  const int num_devices = 1;
  const int batchsize = 1024;
  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz * slot_num, max_nnz,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  constexpr size_t buffer_length = max_nnz;
  std::shared_ptr<HeapEx<CSRChunk<T>>> csr_heap(
      new HeapEx<CSRChunk<T>>(1, num_devices, batchsize, label_dim + dense_dim, params));

  std::vector<long long> slot_offset(slot_size.size(), 0);
  for (unsigned int i = 1; i < slot_size.size(); i++) {
    slot_offset[i] = slot_offset[i-1] + slot_size[i-1];
  }

  // setup a data reader
  ParquetDataReaderWorker<T> data_reader(0, 1, csr_heap, file_list_name, buffer_length,
                                          params, slot_offset, 0, gpu_resource_group);

  // call read a batch
  data_reader.read_a_batch();

  rmm::mr::set_current_device_resource(p_mr);
}

TEST(data_reader_group_test, data_reader_parquet_distributed_test) {
  auto p_mr = rmm::mr::get_current_device_resource();
  generate_parquet_input_files(4, 2048);

  const int batchsize = 1024;
  int numprocs = 1;
  std::vector<int> device_list = {0, 1, 2, 3};
  std::vector<std::vector<int>> vvgpu;
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto& resource_manager = ResourceManagerExt::create(vvgpu, 0);
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, max_nnz * slot_num, max_nnz,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::vector<long long> slot_offset(slot_size.size(), 0);
  for (unsigned int i = 1; i < slot_size.size(); i++) {
    slot_offset[i] = slot_offset[i-1] + slot_size[i-1];
  }

  DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager, true, device_list.size(), false, 0);
  data_reader.create_drwg_parquet(file_list_name, slot_offset, true);
  data_reader.read_a_batch_to_device();
  data_reader.read_a_batch_to_device();

  rmm::mr::set_current_device_resource(p_mr);
}

TEST(data_reader_group_test, data_reader_parquet_localized_test) {
  auto p_mr = rmm::mr::get_current_device_resource();
  generate_parquet_input_files(4, 2048);

  const int batchsize = 1024;
  int numprocs = 1;
  std::vector<int> device_list = {0, 1, 2, 3};
  std::vector<std::vector<int>> vvgpu;
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto& resource_manager = ResourceManagerExt::create(vvgpu, 0);
  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz * slot_num, max_nnz,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::vector<long long> slot_offset(slot_size.size(), 0);
  for (unsigned int i = 1; i < slot_size.size(); i++) {
    slot_offset[i] = slot_offset[i-1] + slot_size[i-1];
  }

  DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager, true, device_list.size(), false, 0);
  data_reader.create_drwg_parquet(file_list_name, slot_offset, true);
  data_reader.read_a_batch_to_device();
  data_reader.read_a_batch_to_device();
  data_reader.read_a_batch_to_device();

  rmm::mr::set_current_device_resource(p_mr);
}

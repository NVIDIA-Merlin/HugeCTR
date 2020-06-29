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

#include <fstream>
#include <thread>
#include "HugeCTR/include/data_reader.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

const std::vector<long long> slot_size = {
    39884406, 39043,    17289,    7420,     20263,  3,     7120, 1543, 63,
    38532951, 2953546,  403346,   10,       2208,   11938, 155,  4,    976,
    14,       39979771, 25641295, 39664984, 585935, 12972, 108,  36};

const std::vector<long long> slot_offset = {
    0,        39884406, 39923449,  39940738,  39948158,  39968421,  39968424,  39975544, 39977087,
    39977150, 78510101, 81463647,  81866993,  81867003,  81869211,  81881149,  81881304, 81881308,
    81882284, 81882298, 121862069, 147503364, 187168348, 187754283, 187767255, 187767363};

const long long num_samples = 131072 * 12;
const int max_nnz = 1;
const int slot_num = 26;
const int label_dim = 1;
const int dense_dim = 13;
typedef unsigned int T;
const Check_t CHK = Check_t::None;
const std::string file_name = "./train_data.bin";
// const std::string file_name = "/etc/workspace/dataset/criteo/raw/dlrm_40m.limit/train_data.bin";
TEST(data_reader_raw, data_reader_worker_raw_test) {
  test::mpi_init();

  // data generation
  data_generation_for_raw(file_name, num_samples, label_dim, dense_dim, slot_num);
  // setup a CSR heap
  // setup a CSR heap
  const int num_devices = 1;
  const int batchsize = 2048;
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, max_nnz * slot_num, 1,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  constexpr size_t buffer_length = max_nnz;
  std::shared_ptr<HeapEx<CSRChunk<T>>> csr_heap(
      new HeapEx<CSRChunk<T>>(1, num_devices, batchsize, label_dim + dense_dim, params));

  // setup a data reader
  // DataReaderWorkerRaw<T> data_reader(0,1,batchsize, (label_dim + dense_dim +
  // slot_num)*sizeof(int), csr_heap, 					file_name, num_samples, params, slot_offset, label_dim);

  auto file_offset_list = std::make_shared<MmapOffsetList>(
      file_name, num_samples, (label_dim + dense_dim + slot_num) * sizeof(int), batchsize, false,
      1);

  DataReaderWorkerRaw<T> data_reader(0, 1, file_offset_list, csr_heap, file_name, params,
                                     slot_offset, label_dim);

  // call read a batch
  data_reader.read_a_batch();
}

TEST(data_reader_raw, data_reader_raw_test) {
  // data generation
  // data_generation_for_raw(file_name, num_samples, label_dim, dense_dim, slot_num);

  const int batchsize = 131072;
  int numprocs = 1, pid = 0;
  std::vector<std::vector<int>> vvgpu;
  std::vector<int> device_list = {0, 1};
#ifdef ENABLE_MPI
  test::mpi_init();
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  auto device_map = std::make_shared<DeviceMap>(vvgpu, pid);
  auto gpu_resource_group = std::make_shared<GPUResourceGroup>(device_map);

  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz * slot_num, 1,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  DataReader<T> data_reader(file_name, batchsize, label_dim, dense_dim, CHK, params,
                            gpu_resource_group, 1, DataReaderType_t::Raw, num_samples, slot_offset);

  long long current_batchsize = data_reader.read_a_batch_to_device();
  std::cout << "current_batchsize: " << current_batchsize << std::endl;
  print_tensor(*data_reader.get_label_tensors()[1], 0, 30);
  print_tensor(*data_reader.get_value_tensors()[1], 0, 30);
  print_tensor(*data_reader.get_row_offsets_tensors()[1], 0, 30);
  print_tensor(*data_reader.get_label_tensors()[0], 0, 30);
  print_tensor(*data_reader.get_dense_tensors()[0], 0, 30);
  print_tensor(*data_reader.get_value_tensors()[0], 0, 30);
  print_tensor(*data_reader.get_row_offsets_tensors()[0], 0, 30);

  current_batchsize = data_reader.read_a_batch_to_device();
  std::cout << "current_batchsize: " << current_batchsize << std::endl;
  print_tensor(*data_reader.get_label_tensors()[1], -10, -1);
  print_tensor(*data_reader.get_value_tensors()[1], 0, 10);
  print_tensor(*data_reader.get_row_offsets_tensors()[1], 0, 10);
  current_batchsize = data_reader.read_a_batch_to_device();
  print_tensor(*data_reader.get_value_tensors()[0], -30, -1);
  print_tensor(*data_reader.get_row_offsets_tensors()[0], -30, -1);
  print_tensor(*data_reader.get_value_tensors()[1], -30, -1);
  print_tensor(*data_reader.get_row_offsets_tensors()[1], -30, -1);

  std::cout << "current_batchsize: " << current_batchsize << std::endl;
  print_tensor(*data_reader.get_label_tensors()[1], -10, -1);
  print_tensor(*data_reader.get_value_tensors()[1], 0, 10);
  print_tensor(*data_reader.get_row_offsets_tensors()[1], 0, 10);
}

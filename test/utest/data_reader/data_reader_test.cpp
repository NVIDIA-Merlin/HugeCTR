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

#include "HugeCTR/include/data_readers/data_reader.hpp"

#include <fstream>
#include <thread>

#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/data_readers/data_reader_worker.hpp"
#include "HugeCTR/include/data_readers/file_list.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

// configuration
const int num_files = 20;
const long long label_dim = 2;
const long long dense_dim = 64;
const long long slot_num = 10;
const long long num_records = 2048 * 2;
const int max_nnz = 30;
typedef long long T;
const int vocabulary_size = 511;
// configurations for data_reader_worker
const std::string file_list_name("data_reader_file_list.txt");
const std::string prefix("./data_reader_test_data/temp_dataset_");
const Check_t CHK = Check_t::Sum;

TEST(data_reader_worker, data_reader_worker_test) {
  test::mpi_init();
  // data generation
  HugeCTR::data_generation_for_test<T, CHK>(file_list_name, prefix, num_files, num_records,
                                            slot_num, vocabulary_size, label_dim, dense_dim,
                                            max_nnz);

  // setup a CSR heap
  const int num_devices = 1;
  const int batchsize = 2048;
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, max_nnz * slot_num, max_nnz,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  constexpr size_t buffer_length = max_nnz;
  std::shared_ptr<HeapEx<CSRChunk<T>>> csr_heap(
      new HeapEx<CSRChunk<T>>(1, num_devices, batchsize, label_dim + dense_dim, params));

  // setup a data reader
  DataReaderWorker<T> data_reader(0, 1, csr_heap, file_list_name, buffer_length, true, CHK, params);

  // call read a batch
  data_reader.read_a_batch();
}

TEST(data_reader_test, data_reader_simple_test) {
  const int batchsize = 2048;

  cudaSetDevice(0);
  test::mpi_init();
  int numprocs = 1;
#ifdef ENABLE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif

  std::vector<std::vector<int>> vvgpu;
  std::vector<int> device_list = {0, 1};
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto& resource_manager = ResourceManager::create(vvgpu, 0);
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, max_nnz * slot_num, max_nnz,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager, true, 12);

  data_reader.create_drwg_norm(file_list_name, CHK);

  data_reader.read_a_batch_to_device();
  /*   print_tensor(data_reader.get_label_tensors()[1], -10, -1);
    print_tensor(data_reader.get_value_tensors()[1], 0, 10);
    print_tensor(data_reader.get_row_offsets_tensors()[1], 0, 10); */
  data_reader.read_a_batch_to_device();
  /*   print_tensor(data_reader.get_label_tensors()[1], -10, -1);
    print_tensor(data_reader.get_value_tensors()[1], 0, 10);
    print_tensor(data_reader.get_row_offsets_tensors()[1], 0, 10); */
  data_reader.read_a_batch_to_device();
  /*   print_tensor(data_reader.get_label_tensors()[1], -10, -1);
    print_tensor(data_reader.get_value_tensors()[1], 0, 10);
    print_tensor(data_reader.get_row_offsets_tensors()[1], 0, 10); */
}

TEST(data_reader_test, data_reader_localized_test) {
  const int batchsize = 2048;

  cudaSetDevice(0);
  test::mpi_init();
  int numprocs = 1;
#ifdef ENABLE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif

  std::vector<std::vector<int>> vvgpu;
  std::vector<int> device_list = {0, 1, 2, 3, 4, 5, 6, 7};
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto& resource_manager = ResourceManager::create(vvgpu, 0);

  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz * slot_num, max_nnz,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  HugeCTR::DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager,
                                     true, 1);

  data_reader.create_drwg_norm(file_list_name, CHK);

  data_reader.read_a_batch_to_device();
  /*   print_tensor(data_reader.get_label_tensors()[1], -10, -1);
    print_tensor(data_reader.get_value_tensors()[1], 0, 10);
    print_tensor(data_reader.get_row_offsets_tensors()[1], 0, 10); */

  data_reader.read_a_batch_to_device();
  /*   print_tensor(data_reader.get_label_tensors()[1], -10, -1);
    print_tensor(data_reader.get_value_tensors()[1], 0, 10);
    print_tensor(data_reader.get_row_offsets_tensors()[1], 0, 10); */

  data_reader.read_a_batch_to_device();
  /*   print_tensor(data_reader.get_label_tensors()[1], -10, -1);
    print_tensor(data_reader.get_value_tensors()[1], 0, 10);
    print_tensor(data_reader.get_row_offsets_tensors()[1], 0, 10); */
}

TEST(data_reader_test, data_reader_mixed_test) {
  const int batchsize = 2048;

  cudaSetDevice(0);
  test::mpi_init();
  int numprocs = 1;
#ifdef ENABLE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif

  std::vector<std::vector<int>> vvgpu;
  std::vector<int> device_list = {0, 1, 2, 3};
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto& resource_manager = ResourceManager::create(vvgpu, 0);
  const DataReaderSparseParam param_localized = {DataReaderSparse_t::Localized,
                                                 max_nnz * (slot_num - 5), max_nnz, slot_num - 5};
  const DataReaderSparseParam param_distributed = {DataReaderSparse_t::Distributed, max_nnz * 5,
                                                   max_nnz, 5};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param_localized);
  params.push_back(param_distributed);

  DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager, true, 1,
                            true);

  data_reader.create_drwg_norm(file_list_name, CHK);

  data_reader.read_a_batch_to_device();
  /*   print_tensor(data_reader.get_label_tensors()[1], -10, -1);
    print_tensor(data_reader.get_value_tensors()[1], 0, 10);
    print_tensor(data_reader.get_row_offsets_tensors()[1], 0, 10); */

  data_reader.read_a_batch_to_device();
  /*   print_tensor(data_reader.get_label_tensors()[1], -10, -1);
    print_tensor(data_reader.get_value_tensors()[1], 0, 10);
    print_tensor(data_reader.get_row_offsets_tensors()[1], 0, 10); */

  data_reader.read_a_batch_to_device();
  /*   print_tensor(data_reader.get_label_tensors()[1], -10, -1);
    print_tensor(data_reader.get_value_tensors()[1], 0, 10);
    print_tensor(data_reader.get_row_offsets_tensors()[1], 0, 10); */
}

#ifdef ENABLE_MPI
TEST(data_reader_test, two_nodes_localized) {
  int batchsize = 2048;
  int numprocs = 1, pid = 0;
  HugeCTR::data_generation_for_test<T, CHK>(file_list_name, prefix, num_files, num_records,
                                            slot_num, vocabulary_size, label_dim, dense_dim,
                                            max_nnz);

  test::mpi_init();
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  if (numprocs != 2) {
    std::cout << "numprocs != 2" << std::endl;
    ASSERT_TRUE(false);
  }

  {
    std::cout << "two Nodes 4 GPUs first batch\n" << std::endl;
    // vvgpu suppose you have at least 4 gpus for each node
    std::vector<std::vector<int>> vvgpu;
    std::vector<int> device_list_0 = {0, 1};
    std::vector<int> device_list_1 = {0, 3};
    vvgpu.push_back(device_list_0);
    vvgpu.push_back(device_list_1);

    auto resource_manager = ResourceManager::create(vvgpu, 0);
    const DataReaderSparseParam param_localized = {DataReaderSparse_t::Localized,
                                                   max_nnz * (slot_num), max_nnz, slot_num};
    std::vector<DataReaderSparseParam> params;
    params.push_back(param_localized);

    DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager, true, 1);

    data_reader.create_drwg_norm(file_list_name, CHK);

    data_reader.read_a_batch_to_device();
    /*     print_tensor(*data_reader.get_label_tensors()[1], -10, -1);
        print_tensor(*dynamic_tensor_cast<float>(data_reader.get_dense_tensors()[1]), -10, -1);
        print_tensor(*data_reader.get_value_tensors()[1], 0, 10);
        print_tensor(*data_reader.get_row_offsets_tensors()[1], 0, 10); */

    std::cout << "two Nodes 4 GPUs second batch\n" << std::endl;
    data_reader.read_a_batch_to_device();
    /*     print_tensor(*data_reader.get_label_tensors()[1], -10, -1);
        print_tensor(*dynamic_tensor_cast<float>(data_reader.get_dense_tensors()[1]), -10, -1);
        print_tensor(*data_reader.get_value_tensors()[1], 0, 10);
        print_tensor(*data_reader.get_row_offsets_tensors()[1], 0, 10); */
  }
  std::cout << "Single Node 4 GPUs\n" << std::endl;
  if (pid == 0) {
    std::vector<std::vector<int>> vvgpu;
    std::vector<int> device_list_0 = {0, 1, 2, 3};
    vvgpu.push_back(device_list_0);

    auto resource_manager = ResourceManager::create(vvgpu, 0);
    const DataReaderSparseParam param_localized = {DataReaderSparse_t::Localized,
                                                   max_nnz * (slot_num), max_nnz, slot_num};
    std::vector<DataReaderSparseParam> params;
    params.push_back(param_localized);

    DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager, true, 1);

    data_reader.create_drwg_norm(file_list_name, CHK);

    data_reader.read_a_batch_to_device();
    /*     print_tensor(*data_reader.get_label_tensors()[1], -10, -1);
        print_tensor(*dynamic_tensor_cast<float>(data_reader.get_dense_tensors()[1]), -10, -1);
        print_tensor(*data_reader.get_value_tensors()[1], 0, 10);
        print_tensor(*data_reader.get_row_offsets_tensors()[1], 0, 10);

        print_tensor(*data_reader.get_label_tensors()[3], -10, -1);
        print_tensor(*dynamic_tensor_cast<float>(data_reader.get_dense_tensors()[1]), -10, -1);
        print_tensor(*data_reader.get_value_tensors()[3], 0, 10);
        print_tensor(*data_reader.get_row_offsets_tensors()[3], 0, 10); */
  }
}
#endif

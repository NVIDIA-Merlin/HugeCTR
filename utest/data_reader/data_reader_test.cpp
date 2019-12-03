/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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


#include "HugeCTR/include/data_reader.hpp"
#include <fstream>
#include <thread>
#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/data_reader_worker.hpp"
#include "HugeCTR/include/file_list.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

// configuration
const std::string file_list_name("sample_file_list.txt");
const int num_files = 20;
const long long label_dim = 2;
const long long slot_num = 10;
const long long num_records = 2048 * 2;
const int max_nnz = 30;
typedef long long T;
const std::string prefix("./data_reader_test_data/temp_dataset_");
const int vocabulary_size = 511;



TEST(data_reader_multi_threads, data_reader_single_thread_test) {
  // writing data
  test::mpi_init();
  HugeCTR::data_generation<T>(file_list_name, prefix, num_files, num_records, slot_num,
                              vocabulary_size, label_dim, max_nnz);

  // setup a file list
  FileList file_list(file_list_name);
  // setup a CSR heap
  const int num_devices = 1;
  const int batchsize = 2048;
  const int max_value_size = max_nnz * batchsize * slot_num;

  constexpr size_t buffer_length = max_nnz;
  CSRChunk<T> chunk(num_devices, batchsize, label_dim, slot_num, max_value_size);
  Heap<CSRChunk<T>> csr_heap(32, chunk);
  // setup a data reader
  DataReaderWorker<T> data_reader(csr_heap, file_list, buffer_length);
  // call read a batch
  data_reader.read_a_batch();
}

template <typename T>
void threadFunc(Heap<CSRChunk<T>>* csr_heap, FileList* file_list, size_t buffer_length) {
  DataReaderWorker<T> data_reader(*csr_heap, *file_list, buffer_length);
  data_reader.read_a_batch();
}

TEST(data_reader_multi_threads, data_reader_multi_threads_test) {
  test::mpi_init();

  // setup a file list
  FileList file_list(file_list_name);
  // setup a CSR heap
  const int num_devices = 1;
  const int batchsize = 2048;
  const int max_value_size = max_nnz * batchsize * slot_num;

  constexpr size_t buffer_length = max_nnz;
  CSRChunk<T> chunk(num_devices, batchsize, label_dim, slot_num, max_value_size);
  Heap<CSRChunk<T>> csr_heap(32, chunk);

  // setup several data readers
  const int num_threads = 5;

  std::vector<std::thread*> data_reader_threads;

  for (int i = 0; i < num_threads; i++) {
    data_reader_threads.push_back(
        new std::thread(threadFunc<T>, &csr_heap, &file_list, buffer_length));
  }

  for (auto th : data_reader_threads) {
    th->join();
  }
}


#if 0
TEST(data_reader_test, data_reader_simple_test) {
  const int batchsize = 2048;
  int numprocs = 1, pid = 0;
  std::vector<std::vector<int>> vvgpu;
  std::vector<int> device_list = {0, 1};
  cudaSetDevice(0);
#ifdef ENABLE_MPI
  test::mpi_init();
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  DeviceMap device_map(vvgpu, pid);
  GPUResourceGroup gpu_resource_group(device_map);

  DataReader<T> data_reader(file_list_name, batchsize, label_dim, slot_num, max_nnz * slot_num,
                            gpu_resource_group, 31, 1);

  data_reader.read_a_batch_to_device();
  print_tensor(*data_reader.get_label_tensors()[1], -10, -1);
  print_tensor(*data_reader.get_value_tensors()[1], 0, 10);
  print_tensor(*data_reader.get_row_offsets_tensors()[1], 0, 10);
  data_reader.read_a_batch_to_device();
  print_tensor(*data_reader.get_label_tensors()[1], -10, -1);
  print_tensor(*data_reader.get_value_tensors()[1], 0, 10);
  print_tensor(*data_reader.get_row_offsets_tensors()[1], 0, 10);
  data_reader.read_a_batch_to_device();
  print_tensor(*data_reader.get_label_tensors()[1], -10, -1);
  print_tensor(*data_reader.get_value_tensors()[1], 0, 10);
  print_tensor(*data_reader.get_row_offsets_tensors()[1], 0, 10);
  //  while(1){}

}
#endif

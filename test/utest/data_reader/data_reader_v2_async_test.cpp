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

#include <omp.h>

#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

#include "HugeCTR/include/data_readers/multi_hot/detail/data_reader_impl.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "common.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

void reader_test(std::vector<int> device_list, size_t file_size, size_t batch_size, int num_threads,
                 int num_batches_per_thread, bool shuffle) {
  const std::string fname = "__tmp_test.dat";
  char* ref_data;
  char* read_data;

  HCTR_LIB_THROW(nvmlInit_v2());

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  const auto resource_manager = ResourceManagerExt::create(vvgpu, 424242);

  HCTR_LIB_THROW(cudaMallocHost(&ref_data, file_size));
  HCTR_LIB_THROW(cudaMallocHost(&read_data, file_size));

#pragma omp parallel
  {
    std::mt19937 gen(424242 + omp_get_thread_num());
    // std::uniform_int_distribution<uint8_t> dis(0, 255);
    std::uniform_int_distribution<uint8_t> dis('a', 'z');

#pragma omp for
    for (size_t i = 0; i < file_size; i++) {
      ref_data[i] = dis(gen);
    }
  }

  {
    std::ofstream fout(fname);
    fout.write(ref_data, file_size);
    fout.flush();
  }

  std::cout << "written test data\n";

  MultiHot::FileSource source;
  source.name = fname;
  source.slot_id = 0;
  source.sample_size_bytes = 1;
  std::vector<MultiHot::FileSource> data_files = {source};

  MultiHot::DataReaderImpl data_reader(data_files, resource_manager, batch_size, num_threads,
                                       num_batches_per_thread, shuffle,
                                       false /* schedule uploads */);

  data_reader.start();

  size_t total_sz = 0;
  while (true) {
    const MultiHot::DataReaderImpl::Batch& batch = data_reader.get_batch();

    const size_t slot = 0;

    for (size_t i = 0; i < device_list.size(); ++i) {
      CudaDeviceContext ctx(device_list[i]);

      auto host_data = batch.get_host_data(i, slot);
      auto device_data = batch.get_device_data(i, slot);
      auto size_bytes = batch.get_size_bytes(i, slot);

      std::vector<uint8_t> local_ref_data(ref_data + total_sz, ref_data + total_sz + size_bytes);
      std::vector<uint8_t> local_data(host_data, host_data + size_bytes);
      ASSERT_EQ(local_data, local_ref_data) << "host data differs at offset " << total_sz;

      HCTR_LIB_THROW(
          cudaMemcpy(read_data + total_sz, device_data, size_bytes, cudaMemcpyDeviceToHost));
      data_reader.device_release_last_batch_here(NULL);

      total_sz += size_bytes;
    }

    if (total_sz >= file_size) {
      break;
    }
  }

  ASSERT_EQ(total_sz, file_size);
  for (size_t i = 0; i < std::min(file_size, total_sz); i++) {
    ASSERT_EQ(ref_data[i], read_data[i]) << "Symbols differ at index " << i << " : expected "
                                         << ref_data[i] << " got " << read_data[i];
  }

  cudaFreeHost(ref_data);
  cudaFreeHost(read_data);
}

//   device_list   file_size   batch_size  threads  batches_per_thread shuffle
//
bool shuffle = false;

TEST(reader_test, test1) { reader_test({0}, 20000, 4096, 1, 4, shuffle); }
TEST(reader_test, test2) { reader_test({0}, 100, 20, 2, 2, shuffle); }
TEST(reader_test, test3) { reader_test({0}, 1012, 20, 2, 1, shuffle); }
TEST(reader_test, test4) { reader_test({0}, 1012, 32, 2, 2, shuffle); }
TEST(reader_test, test5) { reader_test({0}, 10120, 32, 2, 2, shuffle); }
TEST(reader_test, test6) { reader_test({0}, 101256, 1000, 2, 4, shuffle); }
TEST(reader_test, test7) { reader_test({0}, 101256, 1000, 2, 4, shuffle); }
TEST(reader_test, test8) { reader_test({0}, 101256, 1000, 2, 4, shuffle); }
TEST(reader_test, test9) { reader_test({0, 1}, 100, 20, 2, 1, shuffle); }
// TEST(reader_test, test10) { reader_test({0, 1}, 101256, 1000, 2, 4, 512, 2, 0); }
// TEST(reader_test, test11) { reader_test({0, 1}, 101256, 1000, 2, 4, 512, 2, 100); }
// TEST(reader_test, test12) { reader_test({0, 1}, 101256, 1000, 2, 4, 512, 2, 1000); }
// TEST(reader_test, test13) { reader_test({0, 1}, 1014252, 14352, 6, 4, 512, 2, 0); }
// TEST(reader_test, test14) { reader_test({0, 1, 2, 3}, 100980, 1980, 4, 4, 512, 2, 1000); }
// TEST(reader_test, test15) { reader_test({0, 1, 2, 3, 4}, 101256, 7616, 8, 4, 512, 2, 0); }
// TEST(reader_test, test16) {
// reader_test({0, 1, 2, 3, 4, 5, 6, 7}, 8012516, 38720, 8, 4, 512, 2, 0);
//}
// TEST(reader_test, test17) {
// reader_test({0, 1, 2, 3, 4, 5, 6, 7}, 8012516, 38720, 16, 4, 512, 2, 0);
//}
// TEST(reader_test, test18) {
// reader_test({0, 1, 2, 3, 4, 5, 6, 7}, 18012516, 38720, 8, 4, 512, 2, 2000);
//}

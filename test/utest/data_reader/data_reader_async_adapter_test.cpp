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

#include "HugeCTR/include/data_readers/async_reader/async_reader_adapter.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/resource_manager.hpp"
#include "common.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;
using namespace HugeCTR::hybrid_embedding;

size_t global_seed = 321654;

template <typename dtype>
void reader_adapter_test(std::vector<int> device_list, size_t batch_size, int num_threads,
                         int batches_per_thread, int label_dim, int dense_dim, int sparse_dim,
                         int num_passes, int seed) {
  const std::string fname = "__tmp_test.dat";

  const bool mixed_precision = true;
  const float epsilon = mixed_precision ? 1e0f : 1e-3f;

  CK_NVML_THROW_(nvmlInit_v2());

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  const auto resource_manager = ResourceManager::create(vvgpu, 424242);

  size_t local_gpu_count = resource_manager->get_local_gpu_count();
  const int sample_dim = label_dim + dense_dim + sparse_dim * (sizeof(dtype) / sizeof(float));
  const size_t num_batches = 13;
  const size_t file_size = num_batches * batch_size * sample_dim;

  std::vector<int> ref_data(file_size);

#pragma omp parallel
  {
    std::mt19937 gen(seed + omp_get_thread_num());
    std::uniform_int_distribution<dtype> dis(10000, 99999);
    std::uniform_real_distribution<float> disf(0.1, 1.1);

#pragma omp for
    for (size_t i = 0; i < num_batches * batch_size; i++) {
      for (int j = 0; j < label_dim; j++) {
        ref_data[i * sample_dim + j] = dis(gen);
      }

      for (int j = 0; j < dense_dim; j++) {
        ref_data[i * sample_dim + label_dim + j] = dis(gen);
      }

      for (int j = 0; j < sparse_dim; j++) {
        auto dtype_ref =
            reinterpret_cast<dtype*>(ref_data.data() + i * sample_dim + label_dim + dense_dim);
        dtype_ref[j] = dis(gen);
      }
    }
  }

  {
    std::ofstream fout(fname);
    fout.write((char*)ref_data.data(), file_size * sizeof(int));
  }

  std::vector<DataReaderSparseParam> params{
      {DataReaderSparse_t::Distributed, sparse_dim, 1, sparse_dim}};
  AsyncReader<dtype> data_reader(fname, batch_size, label_dim, dense_dim, params, true,
                                 resource_manager, num_threads, batches_per_thread, 512000, 2, 512);

  auto label_tensors = bags_to_tensors<float>(data_reader.get_label_tensors());
  auto dense_tensors = bags_to_tensors<__half>(data_reader.get_dense_tensors());
  auto sparse_tensors = bags_to_tensors<dtype>(data_reader.get_value_tensors());

  data_reader.start();

  for (int pass = 0; pass < num_passes; pass++) {
    size_t total_read = 0;
    for (size_t batch = 0; batch < num_batches; batch++) {
      size_t sz = data_reader.read_a_batch_to_device();

      std::vector<size_t> device_batch_offsets(local_gpu_count + 1);
      size_t total_offset = 0;
      for (size_t id = 0; id < local_gpu_count + 1; id++) {
        device_batch_offsets[id] = total_offset;
        if (id < local_gpu_count) {
          total_offset += data_reader.get_current_batchsize_per_device(id);
        }
      }

      //#pragma omp parallel for num_threads(local_gpu_count)
      for (size_t id = 0; id < local_gpu_count; id++) {
        auto device = resource_manager->get_local_gpu(id);
        CudaDeviceContext context(device->get_device_id());

        std::vector<float> labels;
        std::vector<__half> denses;
        std::vector<dtype> sparses;

        download_tensor(labels, label_tensors[id], device->get_stream());
        download_tensor(denses, dense_tensors[id], device->get_stream());
        download_tensor(sparses, sparse_tensors[id], device->get_stream());

        auto cur_ref = ref_data.data() + total_read * sample_dim;

        for (size_t sample = device_batch_offsets[id]; sample < device_batch_offsets[id + 1];
             sample++) {
          for (int j = 0; j < label_dim; j++) {
            ASSERT_EQ((float)cur_ref[sample * sample_dim + j],
                      labels[(sample - device_batch_offsets[id]) * label_dim + j]);
          }

          for (int j = 0; j < dense_dim; j++) {
            ASSERT_NEAR(std::log((double)cur_ref[sample * sample_dim + label_dim + j] + 1.0),
                        (double)denses[(sample - device_batch_offsets[id]) * dense_dim + j],
                        epsilon);
          }
        }

        for (size_t sample = 0; sample < sz; sample++) {
          for (int j = 0; j < sparse_dim; j++) {
            auto dtype_ref =
                reinterpret_cast<dtype*>(cur_ref + sample * sample_dim + label_dim + dense_dim);
            ASSERT_EQ(dtype_ref[j], sparses[sample * sparse_dim + j]);
          }
        }
      }

      total_read += sz;
    }
  }
}

class MPIEnvironment : public ::testing::Environment {
 protected:
  virtual void SetUp() { test::mpi_init(); }
  virtual void TearDown() { test::mpi_finalize(); }
  virtual ~MPIEnvironment(){};
};

::testing::Environment* const mpi_env = ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

//   device_list   batch  threads  batch_per_thread   label  dense  sparse  num_passes  seed
//
TEST(reader_adapter_test, test1) {
  reader_adapter_test<uint32_t>({0}, 100, 1, 1, 2, 1, 1, 1, global_seed += 128);
}
TEST(reader_adapter_test, test2) {
  reader_adapter_test<uint32_t>({0}, 100, 1, 1, 2, 1, 1, 2, global_seed += 128);
}
TEST(reader_adapter_test, test3) {
  reader_adapter_test<uint32_t>({0}, 100, 1, 1, 2, 3, 1, 3, global_seed += 128);
}
TEST(reader_adapter_test, test4) {
  reader_adapter_test<uint32_t>({0}, 100, 1, 1, 2, 3, 6, 7, global_seed += 128);
}
TEST(reader_adapter_test, test5) {
  reader_adapter_test<uint32_t>({0}, 1012, 2, 1, 2, 3, 7, 18, global_seed += 128);
}
TEST(reader_adapter_test, test6) {
  reader_adapter_test<uint32_t>({0}, 101256, 2, 1, 2, 3, 7, 8, global_seed += 128);
}
TEST(reader_adapter_test, test7) {
  reader_adapter_test<uint32_t>({0}, 101256, 2, 4, 2, 3, 7, 5, global_seed += 128);
}
TEST(reader_adapter_test, test8) {
  reader_adapter_test<uint32_t>({0}, 101256, 2, 3, 3, 3, 9, 2, global_seed += 128);
}
TEST(reader_adapter_test, test9) {
  reader_adapter_test<uint32_t>({0}, 101256, 4, 4, 1, 8, 6, 4, global_seed += 128);
}
TEST(reader_adapter_test, test10) {
  reader_adapter_test<uint32_t>({0, 1}, 10, 2, 2, 7, 2, 1, 21, global_seed += 128);
}
TEST(reader_adapter_test, test11) {
  reader_adapter_test<uint32_t>({0, 1, 2, 3}, 1014252, 3, 2, 7, 13, 26, 1, global_seed += 128);
}

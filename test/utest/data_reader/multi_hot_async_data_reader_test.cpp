/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <omp.h>

#include <common.hpp>
#include <cstdio>
#include <data_readers/multi_hot/async_data_reader.hpp>
#include <embeddings/hybrid_embedding/utils.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <general_buffer2.hpp>
#include <iostream>
#include <resource_managers/resource_manager_ext.hpp>
#include <sstream>
#include <type_traits>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;
using namespace HugeCTR::MultiHot;
using namespace HugeCTR::hybrid_embedding;

size_t global_seed = 321654;
size_t num_batches = 13;

template <typename dtype, bool NewReader = false>
void async_data_reader_test(std::vector<int> device_list, size_t batch_size,
                            int num_threads_per_device, int batches_per_thread, int label_dim,
                            int dense_dim, int sparse_dim, int num_passes, int seed,
                            bool incomplete_batch = false, bool schedule_uploads = false,
                            bool shuffle = false) {
  srand(seed);
  HCTR_LIB_THROW(nvmlInit_v2());

  const std::string fname = "__tmp_test.dat";

  HCTR_LOG_S(INFO, ROOT) << "AsyncDataReader: num_threads_per_device = " << num_threads_per_device
                         << std::endl;
  HCTR_LOG_S(INFO, ROOT) << "AsyncDataReader: num_batches_per_thread = " << batches_per_thread
                         << std::endl;
  HCTR_LOG_S(INFO, ROOT) << "AsyncDataReader: shuffle = " << (shuffle ? "ON" : "OFF") << std::endl;

  const bool mixed_precision = true;
  const float epsilon = mixed_precision ? 1e0f : 1e-3f;

  size_t total_sparse_dim = 0;
  std::vector<int> multi_hot_sizes;
  for (int i = 0; i < sparse_dim; ++i) {
    int hotness = (rand() % 3) + 1;
    multi_hot_sizes.push_back(hotness);
    total_sparse_dim += hotness;
  }

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  const auto resource_manager = ResourceManagerExt::create(vvgpu, 424242);

  size_t local_gpu_count = resource_manager->get_local_gpu_count();
  const int sample_dim = label_dim + dense_dim + (total_sparse_dim * (sizeof(dtype) / sizeof(int)));
  size_t file_size = num_batches * batch_size * sample_dim;

  std::vector<int> ref_data(file_size);

  {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(10000, 99999);
    std::uniform_real_distribution<float> disf(0.1, 1.1);

    std::vector<dtype> tmp_sparse_sample(total_sparse_dim);

    for (size_t i = 0; i < num_batches * batch_size; i++) {
      for (int j = 0; j < label_dim; j++) {
        ref_data[i * sample_dim + j] = dis(gen);
      }

      for (int j = 0; j < dense_dim; j++) {
        ref_data[i * sample_dim + label_dim + j] = dis(gen);
      }

      for (int j = 0; j < total_sparse_dim; j++) {
        tmp_sparse_sample[j] = dis(gen);
      }
      std::memcpy(ref_data.data() + i * sample_dim + label_dim + dense_dim,
                  tmp_sparse_sample.data(), tmp_sparse_sample.size() * sizeof(dtype));
    }
  }

  {
    std::ofstream fout(fname);
    fout.write((char*)ref_data.data(), file_size * sizeof(int));
    fout.flush();
    fout.close();
  }

  if (incomplete_batch) {
    // resize file such that last batch will have 1/2 a batch
    auto p = std::filesystem::current_path() / fname;
    size_t new_file_size = (file_size - ((batch_size / 2) * sample_dim)) * sizeof(int);
    std::filesystem::resize_file(p, new_file_size);
  }

  std::vector<DataReaderSparseParam> params{
      DataReaderSparseParam("dummy", multi_hot_sizes, true, sparse_dim)};

  FileSource source;
  source.name = fname;
  source.slot_id = 0;

  bool is_dense_float = false;
  using DataReaderType = typename std::conditional<NewReader, core23_reader::AsyncDataReader<dtype>,
                                                   AsyncDataReader<dtype>>::type;
  DataReaderType data_reader({source}, resource_manager, batch_size, num_threads_per_device,
                             batches_per_thread, params, label_dim, dense_dim, mixed_precision,
                             shuffle, schedule_uploads, is_dense_float);

  auto label_tensors = bags_to_tensors<float>(data_reader.get_label_tensors());
  auto dense_tensors = bags_to_tensors<__half>(data_reader.get_dense_tensors());

  data_reader.start();
  for (int pass = 0; pass < 0; pass++) {
    size_t total_read = 0;
    for (size_t batch = 0; batch < num_batches; batch++) {
      size_t current_batch_size = data_reader.read_a_batch_to_device();

      auto sparse_tensors = data_reader.get_current_sparse_tensors();
      std::cout << batch << " batch good:" << current_batch_size << std::endl;

      std::vector<size_t> device_batch_offsets(local_gpu_count + 1);
      size_t total_offset = 0;
      for (size_t id = 0; id < local_gpu_count + 1; id++) {
        device_batch_offsets[id] = total_offset;
        if (id < local_gpu_count) {
          auto batch_size_per_dev = data_reader.get_current_batchsize_per_device(id);
          total_offset += batch_size_per_dev;

          const bool final_batch = batch == (num_batches - 1);
          if (incomplete_batch && final_batch) {
            // assert first half of devices have batch and second half have no batch
            if (id < local_gpu_count / 2) {
              ASSERT_EQ(batch_size_per_dev, batch_size / local_gpu_count);
            } else {
              ASSERT_EQ(batch_size_per_dev, 0);
            }
          }
        }
      }

      for (size_t id = 0; id < local_gpu_count; id++) {
        auto device = resource_manager->get_local_gpu(id);
        CudaDeviceContext context(device->get_device_id());

        std::vector<float> labels;
        std::vector<__half> denses;
        std::vector<std::vector<dtype>> sparses(sparse_dim);

        download_tensor(labels, label_tensors[id], device->get_stream());
        download_tensor(denses, dense_tensors[id], device->get_stream());
        for (int feat_id = 0; feat_id < sparse_dim; ++feat_id) {
          download_tensor(sparses[feat_id], sparse_tensors[id][feat_id].get_value_tensor(),
                          device->get_stream());
        }

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

          int offset = 0;
          for (int j = 0; j < sparse_dim; ++j) {
            std::vector<dtype> ref_sparse(multi_hot_sizes[j]);
            std::memcpy(ref_sparse.data(),
                        &cur_ref[sample * sample_dim + label_dim + dense_dim + offset],
                        multi_hot_sizes[j] * sizeof(dtype));

            for (int k = 0; k < multi_hot_sizes[j]; ++k) {
              ASSERT_EQ(ref_sparse[k],
                        sparses[j][(sample - device_batch_offsets[id]) * multi_hot_sizes[j] + k]);
              offset += (sizeof(dtype) / sizeof(int));
            }
          }
        }
      }

      total_read += current_batch_size;
    }
  }
}

// class MPIEnvironment : public ::testing::Environment {
// protected:
//  virtual void SetUp() { test::mpi_init(); }
//  virtual void TearDown() { test::mpi_finalize(); }
//  virtual ~MPIEnvironment(){};
//};

//::testing::Environment* const mpi_env = ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

//   device_list   batch  threads  batch_per_thread   label  dense  sparse  num_passes  seed

TEST(async_data_reader_test, gpu_1x_basic) {
  async_data_reader_test<uint32_t, false>({0}, 100, 1, 1, 2, 3, 5, 1, global_seed += 128);
  async_data_reader_test<uint32_t, true>({0}, 100, 1, 1, 2, 3, 5, 1, global_seed += 128);
}
TEST(async_data_reader_test, gpu_1x_basic_long_long) {
  async_data_reader_test<long long, false>({0}, 100, 1, 1, 2, 3, 5, 1, global_seed += 128);
  async_data_reader_test<long long, true>({0}, 100, 1, 1, 2, 3, 5, 1, global_seed += 128);
}
TEST(async_data_reader_test, gpu_1x_multi_threaded) {
  async_data_reader_test<uint32_t, false>({0}, 100, 4, 1, 2, 3, 5, 1, global_seed += 128);
  async_data_reader_test<uint32_t, true>({0}, 100, 4, 1, 2, 3, 5, 1, global_seed += 128);
}
TEST(async_data_reader_test, gpu_1x_multiple_batches_per_thread) {
  async_data_reader_test<uint32_t, false>({0}, 100, 4, 4, 2, 3, 5, 1, global_seed += 128);
  async_data_reader_test<uint32_t, true>({0}, 100, 4, 4, 2, 3, 5, 1, global_seed += 128);
}
TEST(async_data_reader_test, gpu_8x_incomplete_batch) {
  async_data_reader_test<uint32_t, false>({0, 1, 2, 3, 4, 5, 6, 7}, 128, 1, 1, 2, 3, 5, 1,
                                          global_seed += 128, true);
  async_data_reader_test<uint32_t, true>({0, 1, 2, 3, 4, 5, 6, 7}, 128, 1, 1, 2, 3, 5, 1,
                                         global_seed += 128, true);
}
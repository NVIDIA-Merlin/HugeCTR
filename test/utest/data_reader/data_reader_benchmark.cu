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
#include <data_readers/async_reader/async_reader.hpp>
#include <data_readers/multi_hot/detail/data_reader_impl.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <general_buffer2.hpp>
#include <iostream>
#include <resource_managers/resource_manager_ext.hpp>
#include <sstream>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

// const std::string fname = "/40m.limit_preshuffled/train_data.bin";
// const std::string fname = "/criteo_kaggle/dlrm/test_data.bin";
// const std::string fname = "/raid/datasets/criteo/mlperf/40m.limit_preshuffled/train_data.bin";
const std::string fname = "/data_multihot/val_data.bin";

__global__ void gpu_sleep(int64_t num_cycles) {
  int64_t cycles = 0;
  int64_t start = clock64();
  while (cycles < num_cycles) {
    cycles = clock64() - start;
  }
}

int64_t get_cycles(float seconds) {
  // Get device frequency in KHz
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int64_t Hz = int64_t(prop.clockRate) * 1000;

  // Calculate number of cycles to wait
  int64_t num_cycles = (int64_t)(seconds * Hz);

  return num_cycles;
}

void reader_test(std::vector<int> device_list, size_t batch_size_bytes, int num_threads,
                 int num_batches_per_thread, bool shuffle) {
  size_t file_size = batch_size_bytes * 200;  // 1e9 * 100;//std::filesystem::file_size(fname);

  std::cout << "file_size: " << file_size << std::endl;
  std::cout << "batch_size_bytes: " << batch_size_bytes << std::endl;

  HCTR_LIB_THROW(nvmlInit_v2());

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  const auto resource_manager = ResourceManagerExt::create(vvgpu, 424242);

  MultiHot::FileSource source;
  source.name = fname;
  source.slot_id = 0;
  source.sample_size_bytes = 1;
  std::vector<MultiHot::FileSource> data_files = {source};

  const int64_t cycles_wait = get_cycles(0.01);

  // const std::vector<std::vector<std::string>>& node_sources,
  //                 const std::map<std::string, SourceMetadata>& source_metadata,
  //                 const std::shared_ptr<ResourceManager>& resource_manager,
  //                 size_t batch_size,
  //                 size_t num_threads_per_source,
  //                 size_t num_batches_per_thread
  //

  // AsyncReaderImpl reader_impl(fname, batch_size_bytes, resource_manager.get(), num_threads,
  //	                                    num_batches_per_thread, 552960, 2, 512);

  MultiHot::DataReaderImpl data_reader(data_files, resource_manager, batch_size_bytes, num_threads,
                                       num_batches_per_thread, shuffle,
                                       true /* schedule uploads */);

  auto start = std::chrono::high_resolution_clock::now();

  data_reader.start();
  // reader_impl.load_async();
  //

  printf("start\n");

  size_t total_sz = 0;
  while (true) {
    const MultiHot::DataReaderImpl::Batch& batch = data_reader.get_batch();
    total_sz += batch.get_batch_size_bytes();

#pragma omp parallel for num_threads(device_list.size())
    for (size_t i = 0; i < device_list.size(); i++) {
      auto gpu = resource_manager->get_local_gpu(i);
      CudaDeviceContext ctx(gpu->get_device_id());
      data_reader.device_release_last_batch_here(gpu->get_stream());
      gpu_sleep<<<1, 1, 0, gpu->get_stream()>>>(cycles_wait);
      data_reader.schedule_upload_here(i, gpu->get_stream(), false);
      data_reader.upload_notify(i);
    }

    //    std::cout << "total_size: " << total_sz << std::endl;

    /*	  BatchDesc desc = reader_impl.get_batch();
                  size_t sz = desc.size_bytes;
                  if(sz > 0) {
                  total_sz += sz;

                  reader_impl.finalize_batch();
                  }
      */
    if (total_sz >= file_size) {
      break;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  double gbs = ((double)file_size / 1e9) / (double)(time_ms / 1000.f);
  std::cout << "Time (ms): " << time_ms << ",  Throughput (GB/s): " << gbs << std::endl;
}

//   device_list   batch_size   threads  batches_per_thread shuffle
//
const bool shuffle = false;

// TEST(reader_benchmark, train_data) { reader_test({0}, 8847360, 1, 16, shuffle); }
TEST(reader_benchmark, train_data) {
  reader_test({0, 1, 2, 3, 4, 5, 6, 7}, 8847360, 1, 16, shuffle);
}

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

#include <algorithm>
#include <numeric>
#include <random>

#include "HugeCTR/include/shuffle/shuffle.cuh"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

template <typename SrcT, typename DstT>
void shuffle_one2one(int num_elems, int num_dimensions) {
  constexpr float eps = 1e-3;

  assert((size_t)num_elems * num_dimensions == (size_t)(num_elems * num_dimensions));
  SrcT *src;
  DstT *dst;
  int *src_ids, *dst_ids;
  int *src_memcpy, *dst_memcpy;

  size_t src_size = num_elems * num_dimensions * sizeof(SrcT);
  size_t dst_size = num_elems * num_dimensions * sizeof(DstT);
  size_t total_size = src_size + dst_size;
  size_t memcpy_size = (total_size / (2 * sizeof(int))) * sizeof(int);

  cudaMallocManaged((void **)&src, src_size);
  cudaMallocManaged((void **)&dst, dst_size);
  cudaMallocManaged((void **)&src_ids, num_elems * sizeof(int));
  cudaMallocManaged((void **)&dst_ids, num_elems * sizeof(int));
  cudaMalloc((void **)&src_memcpy, memcpy_size);
  cudaMalloc((void **)&dst_memcpy, memcpy_size);

  std::mt19937 gen(4242);
  std::uniform_real_distribution<float> dist(0, 1);
  std::generate(src, src + num_elems * num_dimensions, [&]() { return (SrcT)dist(gen); });
  std::iota(src_ids, src_ids + num_elems, 0);
  std::iota(dst_ids, dst_ids + num_elems, 0);
  std::shuffle(src_ids, src_ids + num_elems, gen);
  std::shuffle(dst_ids, dst_ids + num_elems, gen);

  auto copy_info = CopyDescriptors::make_OneToOne<SrcT, DstT, 1>(
      num_dimensions,
      [=] __device__() { return num_elems; },
      [=] __device__(size_t src_id) -> CopyDescriptors::CopyDetails<SrcT, DstT, 1> {
        return {src + src_ids[src_id] * num_dimensions,
                {dst + dst_ids[src_id] * num_dimensions},
                {true}};
      });

  shuffle(copy_info, (cudaStream_t)0, num_elems);
  cudaDeviceSynchronize();

  for (int i = 0; i < num_elems; i++) {
    for (int j = 0; j < num_dimensions; j++) {
      ASSERT_NEAR(src[src_ids[i] * num_dimensions + j], dst[dst_ids[i] * num_dimensions + j], eps)
          << "Mismatch at element " << i << " dimension " << j << std::endl;
    }
  }

  // Touch on the GPU
  shuffle(copy_info, (cudaStream_t)0, num_elems);
  cudaDeviceSynchronize();

  // Speed test
  float copy_time, kernel_time;
  int niters = round(std::min(200.0, 1e10 / (total_size)));
  cudaStream_t stream;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStreamCreate(&stream);

  // Simple copy
  cudaEventRecord(start, stream);
  for (int i = 0; i < niters; i++) {
    cudaMemcpyAsync(dst_memcpy, src_memcpy, memcpy_size, cudaMemcpyDeviceToDevice, stream);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&copy_time, start, stop);

  // Kernel copy
  cudaEventRecord(start, stream);
  for (int i = 0; i < niters; i++) {
    shuffle(copy_info, stream, num_elems);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernel_time, start, stop);

  printf("Memcpy time: %.3f us (%.1f GB/s), kernel time: %.3f us (%.1f GB/s), %.2f%% of the peak\n",
         1e3 * copy_time / niters, total_size / (copy_time / niters) * 1e-6,
         1e3 * kernel_time / niters, total_size / (kernel_time / niters) * 1e-6,
         copy_time / kernel_time * 100.0);

  cudaFree(src);
  cudaFree(dst);
  cudaFree(src_ids);
  cudaFree(dst_ids);
  cudaFree(src_memcpy);
  cudaFree(dst_memcpy);
}

//
TEST(shuffle_one2one, test1) { shuffle_one2one<float, float>(800, 31); }
TEST(shuffle_one2one, test2) { shuffle_one2one<float, float>(800, 64); }
TEST(shuffle_one2one, test3) { shuffle_one2one<float, float>(8521, 128); }
TEST(shuffle_one2one, test4) { shuffle_one2one<float, float>(16000, 43); }
TEST(shuffle_one2one, test5) { shuffle_one2one<float, float>(16000, 121); }
TEST(shuffle_one2one, test6) { shuffle_one2one<float, float>(16001, 32); }
TEST(shuffle_one2one, test7) { shuffle_one2one<float, float>(16000, 64); }
TEST(shuffle_one2one, test8) { shuffle_one2one<float, float>(16000, 128); }
TEST(shuffle_one2one, test9) { shuffle_one2one<float, float>(160000, 128); }
TEST(shuffle_one2one, test10) { shuffle_one2one<float, float>(730083, 128); }
TEST(shuffle_one2one, test11) { shuffle_one2one<float, __half>(801, 128); }
TEST(shuffle_one2one, test12) { shuffle_one2one<float, __half>(8522, 128); }
TEST(shuffle_one2one, test13) { shuffle_one2one<float, __half>(16013, 64); }
TEST(shuffle_one2one, test14) { shuffle_one2one<float, __half>(170000, 128); }
TEST(shuffle_one2one, test15) { shuffle_one2one<float, __half>(930083, 128); }
TEST(shuffle_one2one, test16) { shuffle_one2one<__half, float>(800, 128); }
TEST(shuffle_one2one, test17) { shuffle_one2one<__half, float>(8521, 128); }
TEST(shuffle_one2one, test18) { shuffle_one2one<__half, float>(16000, 128); }
TEST(shuffle_one2one, test19) { shuffle_one2one<__half, float>(160000, 64); }
TEST(shuffle_one2one, test20) { shuffle_one2one<__half, float>(730083, 128); }
TEST(shuffle_one2one, test21) { shuffle_one2one<__half, float>(830083, 256); }
TEST(shuffle_one2one, test22) { shuffle_one2one<__half, __half>(800, 128); }
TEST(shuffle_one2one, test23) { shuffle_one2one<__half, __half>(85, 34); }
TEST(shuffle_one2one, test24) { shuffle_one2one<__half, __half>(16000, 128); }
TEST(shuffle_one2one, test25) { shuffle_one2one<__half, __half>(160000, 200); }
TEST(shuffle_one2one, test26) { shuffle_one2one<__half, __half>(730083, 128); }
TEST(shuffle_one2one, test27) { shuffle_one2one<__half, __half>(830083, 256); }

/// TODO: test selective copy and multiple destinations

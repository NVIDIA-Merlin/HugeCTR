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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "HugeCTR/embedding/operators/compress_offset.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "HugeCTR/core/hctr_impl/hctr_backend.hpp"
using namespace embedding;

TEST(test_compress_offset, test_compress_offset) {
  Device device{DeviceType::GPU, 0};
  auto resource_manager = HugeCTR::ResourceManagerExt::create({{0}}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);
  CudaDeviceContext context(device.index());
  int batch_size = 5;
  int num_table = 2;
  int num_offset = batch_size * num_table + 1;
  int num_compressed_offset = num_table + 1;

  auto buffer_ptr = core::GetBuffer(core);
  auto offset = buffer_ptr->reserve({num_offset}, device, TensorScalarType::UInt32);
  buffer_ptr->allocate();

  std::vector<uint32_t> cpu_offset{0};
  for (int i = 1; i < num_offset; ++i) {
    int n = rand() % 10;
    cpu_offset.push_back(n);
  }
  std::partial_sum(cpu_offset.begin(), cpu_offset.end(), cpu_offset.begin());
  offset.copy_from(cpu_offset);
  
  CompressOffset compress_offset{core, num_compressed_offset};
  Tensor compressed_offset;
  compress_offset.compute(offset, batch_size, &compressed_offset);

  std::vector<uint32_t> gpu_compressed_offset;
  compressed_offset.to(&gpu_compressed_offset);
  
  ASSERT_EQ(gpu_compressed_offset.size(), num_compressed_offset);
  for (int i = 0; i < num_compressed_offset; ++i) {
    ASSERT_EQ(gpu_compressed_offset[i], cpu_offset[i * batch_size]);
  }
}
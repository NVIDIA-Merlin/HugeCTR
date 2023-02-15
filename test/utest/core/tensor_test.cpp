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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <core/hctr_impl/hctr_backend.hpp>
#include <core/tensor.hpp>
#include <resource_managers/resource_manager_ext.hpp>
#include <utils.hpp>

namespace {
using namespace core;

TEST(test_core, tensor) {
  auto resource_manager = HugeCTR::ResourceManagerExt::create({{0}}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);
  Tensor t;
  {
    BufferPtr buffer_ptr = GetBuffer(core);
    t = buffer_ptr->reserve({6, 4}, DeviceType::CPU, TensorScalarType::Float32);
    t = buffer_ptr->reserve({6, 4}, DeviceType::CPU, TensorScalarType::Float32);
    buffer_ptr->allocate();
  }
  t.get<float>()[0] = 0.f;

  Tensor copy_t;
  {
    BufferPtr buffer_ptr = GetBuffer(core);
    copy_t = buffer_ptr->reserve({6, 4}, DeviceType::GPU, TensorScalarType::Float32);
    buffer_ptr->allocate();
  }
  std::cout << "copy_t device:" << copy_t.device() << std::endl;
  copy_t = t;
  std::cout << "copy_t device:" << copy_t.device() << std::endl;
  Tensor other_copy_t = t;
  EXPECT_TRUE(t.get<float>()[0] == 0.f);
};

TEST(test_core, cpu_copy) {
  auto resource_manager = HugeCTR::ResourceManagerExt::create({{0}}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);

  Tensor cpu_t_1;
  {
    BufferPtr buffer_ptr = GetBuffer(core);
    cpu_t_1 = buffer_ptr->reserve({1}, DeviceType::CPU, TensorScalarType::Float32);
    buffer_ptr->allocate();
  }
  cpu_t_1.get<float>()[0] = 10.0f;
  Tensor cpu_t_2 = cpu_t_1.to(core, DeviceType::CPU);

  EXPECT_TRUE(cpu_t_1.get<float>()[0] == cpu_t_2.get<float>()[0]);
}

TEST(test_core, gpu_copy) {
  auto resource_manager = HugeCTR::ResourceManagerExt::create({{0}}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);

  Tensor cpu_t_1;
  {
    BufferPtr buffer_ptr = GetBuffer(core);
    cpu_t_1 = buffer_ptr->reserve({1}, DeviceType::CPU, TensorScalarType::Float32);
    buffer_ptr->allocate();
  }
  cpu_t_1.get<float>()[0] = 1.0f;
  Tensor gpu_t_1 = cpu_t_1.to(core, DeviceType::GPU);
  Tensor cpu_t_2 = gpu_t_1.to(core, DeviceType::CPU);
  EXPECT_TRUE(cpu_t_1.get<float>()[0] == cpu_t_2.get<float>()[0]);
}

TEST(test_core, buffer_block) {
  auto resource_manager = HugeCTR::ResourceManagerExt::create({{0}}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);

  BufferBlockPtr buffer_block_ptr = GetBufferBlock(core, DeviceType::CPU);
  Tensor char_t = buffer_block_ptr->reserve({1}, TensorScalarType::Char);
  Tensor float_t = buffer_block_ptr->reserve({1}, TensorScalarType::Float32);
  Tensor char_t_2 = buffer_block_ptr->reserve({1}, TensorScalarType::Char);
  buffer_block_ptr->allocate();
  HCTR_LOG(DEBUG, ROOT, "char tensor pointer address: %p\n", char_t.get<char>());
  HCTR_LOG(DEBUG, ROOT, "float tensor pointer address: %p\n", float_t.get<float>());
  EXPECT_TRUE((uintptr_t)char_t.get<char>() % 1 == 0);
  EXPECT_TRUE((uintptr_t)char_t_2.get<char>() % 1 == 0);
  EXPECT_TRUE((uintptr_t)float_t.get<float>() % 4 == 0);
}

// TEST(test_core, cpu_copy_overlap) {
//   Tensor gpu_t_1;
//   Tensor gpu_t_1_copy;
//   Tensor gpu_t_2;
//   Tensor gpu_t_2_copy;
//   {
//     StreamContext stream("back1");
//     gpu_t_1.copy(gpu_t_1_copy, non_blocking=true);
//   }
//   gpu_t_2.copy(gpu_t_2_copy, non_blocking=true);

// }

TEST(test_core, tensor_list) {
  auto resource_manager = HugeCTR::ResourceManagerExt::create({{0}}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);

  std::vector<std::vector<Tensor>> tensors;

  auto buffer_ptr = GetBuffer(core);
  for (int i = 0; i < 2; ++i) {
    HugeCTR::CudaDeviceContext ctx(i);
    std::vector<Tensor> tmp_tensors;
    for (int j = 0; j < 2; ++j) {
      tmp_tensors.push_back(buffer_ptr->reserve({1}, DeviceType::GPU, TensorScalarType::Int32));
    }
    tensors.push_back(tmp_tensors);
  }
  buffer_ptr->allocate();

  std::vector<TensorList> tensor_list;
  for (int i = 0; i < 2; ++i) {
    Device device{DeviceType::GPU, i};
    tensor_list.emplace_back(core.get(), tensors[i], device, TensorScalarType::Int32);
  }
}

TEST(test_core, native_hugectr_tensor) {
  auto resource_manager = HugeCTR::ResourceManagerExt::create({{0}}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);

  auto native_hctr_buffer = HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>::create();
  HugeCTR::Tensor2<float> native_t;
  native_hctr_buffer->reserve({10}, &native_t);
  native_hctr_buffer->allocate();

  Storage storage = std::make_shared<hctr_internal::NativeHCTRStorageWrapper>(
      native_t.get_ptr(), native_t.get_size_in_bytes());
  std::shared_ptr<TensorImpl> t_impl =
      std::make_shared<TensorImpl>(storage, 0, native_t.get_dimensions(), DeviceType::GPU,
                                   HugeCTR::TensorScalarTypeFunc<float>::get_type());
  Tensor t{t_impl};
}

}  // namespace

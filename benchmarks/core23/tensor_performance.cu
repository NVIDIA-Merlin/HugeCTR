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
#include <base/debug/logger.hpp>
#include <chrono>
#include <core23/allocator_params.hpp>
#include <core23/buffer_channel.hpp>
#include <core23/buffer_params.hpp>
#include <core23/data_type.hpp>
#include <core23/details/tensor_impl.hpp>
#include <core23/device.hpp>
#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_params.hpp>
#include <cstdint>
#include <vector>

namespace {

using namespace HugeCTR::core23;

class FakeTensor {
 public:
  FakeTensor(TensorParams params = TensorParams());
  void* data() const;

 private:
  std::shared_ptr<TensorImpl> impl_;
};

FakeTensor::FakeTensor(TensorParams params) : impl_(std::make_shared<TensorImpl>(params)) {}
void* FakeTensor::data() const { return impl_ ? impl_->data() : nullptr; }

const BufferParams g_buffer_params = {};
const AllocatorParams g_allocator_params = {};

template <typename Placeholder>
void value_bench(BufferParams buffer_params, AllocatorParams allocator_params) {
  TensorParams tensor_params = TensorParams()
                                   .data_type(ScalarType::Int32)
                                   .allocator_params(allocator_params)
                                   .buffer_params(buffer_params)
                                   .device(Device(DeviceType::CPU))
                                   .shape({1});

  constexpr size_t num_tensors = 1048576;
  std::vector<Placeholder> tensors;
  tensors.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; i++) {
    tensors.emplace_back(tensor_params);
  }

  uint64_t sum = 0;

  for (const auto& tensor : tensors) {
    sum ^= reinterpret_cast<uint64_t>(tensor.data());
  }

  auto b_t = std::chrono::steady_clock::now();
  for (const auto& tensor : tensors) {
    sum ^= reinterpret_cast<uint64_t>(tensor.data());
  }
  auto e_t = std::chrono::steady_clock::now();
  HCTR_LOG_S(INFO, ROOT) << std::chrono::duration_cast<std::chrono::milliseconds>(e_t - b_t).count()
                         << " ms to get the value " << std::hex << sum << std::endl;
  HCTR_LOG_S(INFO, ROOT) << std::hex << sum << std::endl;
}

template <typename Placeholder>
void pointer_bench(BufferParams buffer_params, AllocatorParams allocator_params) {
  TensorParams tensor_params = TensorParams()
                                   .data_type(ScalarType::Int32)
                                   .allocator_params(allocator_params)
                                   .buffer_params(buffer_params)
                                   .device(Device(DeviceType::GPU, 0))
                                   .shape({32, 32});

  constexpr size_t num_tensors = 1048576;
  std::vector<std::shared_ptr<Placeholder>> tensors;
  tensors.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; i++) {
    tensors.emplace_back(new Placeholder(tensor_params));
  }

  uint64_t sum = 0;

  for (const auto& tensor : tensors) {
    sum += reinterpret_cast<uint64_t>(tensor->data());
  }

  auto b_t = std::chrono::steady_clock::now();
  for (const auto& tensor : tensors) {
    sum += reinterpret_cast<uint64_t>(tensor->data());
  }
  auto e_t = std::chrono::steady_clock::now();
  HCTR_LOG_S(INFO, ROOT) << std::chrono::duration_cast<std::chrono::milliseconds>(e_t - b_t).count()
                         << " ms to get the value " << std::hex << sum << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool call_by_value = true;
    bool with_indirection_overhead = false;
    if (argc >= 2) {
      std::istringstream(std::string(argv[1])) >> call_by_value;
    }
    if (argc >= 3) {
      std::istringstream(std::string(argv[2])) >> with_indirection_overhead;
    }

    AllocatorParams allocator_params = g_allocator_params;
    BufferParams buffer_params = g_buffer_params;

    if (call_by_value) {
      buffer_params.channel = "TENSOR_DATA_PERFORMANCE";
      if (with_indirection_overhead) {
        value_bench<FakeTensor>(buffer_params, allocator_params);
      } else {
        value_bench<Tensor>(buffer_params, allocator_params);
      }
    } else {
      buffer_params.channel = "TENSOR_DATA_PERFORMANCE";
      if (with_indirection_overhead) {
        pointer_bench<FakeTensor>(buffer_params, allocator_params);
      } else {
        pointer_bench<Tensor>(buffer_params, allocator_params);
      }
    }
  } catch (...) {
    HCTR_LOG_S(INFO, ROOT) << "Something is wrong" << std::endl;
  }
}
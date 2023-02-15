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
#pragma once

#include <core/tensor.hpp>

namespace core {

class BufferBlockImpl {
  Storage storage_;
  Device device_;
  std::vector<std::shared_ptr<TensorImpl>> tensors_;

 public:
  BufferBlockImpl(CoreResourceManager *backend, Device device);

  Tensor reserve(const Shape &shape, DataType type);

  void allocate();

  Tensor as_tensor();
};

class BufferImpl final {
  CoreResourceManager *backend_;
  std::unordered_map<Device, Storage> storages_;

 public:
  BufferImpl(CoreResourceManager *backend);

  Tensor reserve(const Shape &shape, Device device, DataType type, size_t alignned = 32ul);

  void allocate();
};

inline BufferBlockPtr GetBufferBlock(std::shared_ptr<CoreResourceManager> backend, Device device) {
  return std::make_shared<core::BufferBlockImpl>(backend.get(), device);
}

inline BufferPtr GetBuffer(std::shared_ptr<CoreResourceManager> backend) {
  return std::make_shared<core::BufferImpl>(backend.get());
}

}  // namespace core
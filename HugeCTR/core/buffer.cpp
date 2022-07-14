/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "buffer.hpp"

#include "core.hpp"

namespace core {

BufferBlockImpl::BufferBlockImpl(CoreResourceManager *backend, Device device)
    : storage_(backend->CreateStorage(device)), device_(device) {}

Tensor BufferBlockImpl::reserve(const Shape &shape, DataType type) {
  auto t = std::make_shared<TensorImpl>(storage_, storage_->nbytes(), shape, device_, type);
  tensors_.push_back(t);
  return Tensor{t};
}

void BufferBlockImpl::allocate() {
  // reorder, move large itemsize type ahead to avoid padding because we need each tensor aligned
  // with its type
  std::sort(tensors_.begin(), tensors_.end(),
            [&](const std::shared_ptr<TensorImpl> &lhs, const std::shared_ptr<TensorImpl> &rhs) {
              return lhs->dtype().itemsize() > rhs->dtype().itemsize();
            });
  size_t bytes_offset = 0;
  for (auto &t : tensors_) {
    t->set_storage_offset(bytes_offset);
    bytes_offset += t->get_dimensions().num_elements() * t->dtype().itemsize();
  }

  HCTR_LOG(DEBUG, ROOT, "internal buffer bytes size: %lu\n", storage_->nbytes());
  storage_->extend(bytes_offset);

  if (device_.is_gpu()) {
    CudaDeviceContext context(device_.index());
    storage_->allocate();
  } else {
    storage_->allocate();
  }
}

// TODO
Tensor BufferBlockImpl::as_tensor() {
  Shape shape{storage_->nbytes()};
  auto t = std::make_shared<TensorImpl>(storage_, 0, shape, device_, TensorScalarType::Char);
  return Tensor{t};
}

BufferImpl::BufferImpl(CoreResourceManager *backend) : backend_(backend) {}

Tensor BufferImpl::reserve(const Shape &shape, Device device, DataType type, size_t alignned) {
  if (storages_.find(device) == storages_.end()) {
    storages_[device] = backend_->CreateStorage(device);
  }
  auto &storage = storages_[device];

  std::shared_ptr<TensorImpl> t =
      std::make_shared<TensorImpl>(storage, storage->nbytes(), shape, device, type);

  size_t current_size_in_bytes = shape.num_elements() * type.itemsize();
  size_t alignned_size_in_bytes = HugeCTR::divup(current_size_in_bytes, alignned) * alignned;
  storage->extend(alignned_size_in_bytes);

  return Tensor{t};
}

void BufferImpl::allocate() {
  for (auto &x : storages_) {
    if (x.first.is_gpu()) {
      CudaDeviceContext context(x.first.index());
      x.second->allocate();
    } else {
      x.second->allocate();
    }
  }
  storages_.clear();
}
}  // namespace core
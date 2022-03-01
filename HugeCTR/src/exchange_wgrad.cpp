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

#include <exchange_wgrad.hpp>
#include <utils.hpp>

namespace HugeCTR {

template <typename T>
NetworkExchangeWgrad<T>::NetworkExchangeWgrad(
    const std::shared_ptr<ResourceManager>& resource_manager)
    : resource_manager_(resource_manager), num_gpus_(resource_manager->get_local_gpu_count()) {
  bufs_.resize(num_gpus_, NULL);
  network_wgrad_buffs_.resize(num_gpus_, NULL);
  null_wgrad_buffs_.resize(num_gpus_, NULL);

  for (size_t g = 0; g < num_gpus_; g++) {
    bufs_[g] = GeneralBuffer2<CudaAllocator>::create();
    network_wgrad_buffs_[g] = bufs_[g]->create_block<T>();
  }

  auto ar_comm = resource_manager_->get_ar_comm();
  ar_handle_ = ar_comm->register_coll();
}

template <typename T>
void NetworkExchangeWgrad<T>::allocate() {
  int alignment = 16 * num_gpus_;
  for (size_t g = 0; g < num_gpus_; g++) {
    auto& gpu_resource = resource_manager_->get_local_gpu(g);
    CudaDeviceContext context(gpu_resource->get_device_id());
    bufs_[g]->allocate_aligned(alignment);
  }

  network_wgrad_size_ = network_wgrad_buffs_[0]->as_tensor().get_size_in_bytes();
  if (network_wgrad_size_ % alignment != 0) {
    network_wgrad_size_ += (alignment - (network_wgrad_size_ % alignment));
  }

  auto ar_comm = resource_manager_->get_ar_comm();
  for (size_t g = 0; g < num_gpus_; g++) {
    ar_comm->set_coll_buf(ar_handle_, bufs_[g]->get_ptr(), network_wgrad_size_, g);
  }
  ar_comm->register_coll_buf(ar_handle_);
}

template <typename T>
void NetworkExchangeWgrad<T>::update_embed_wgrad_size(size_t size) {
  HCTR_OWN_THROW(Error_t::IllegalCall, "Network wgrad exchange can't update embed wgrad size!");
}

template <typename T>
void NetworkExchangeWgrad<T>::allreduce(size_t device_id, cudaStream_t stream) {
  auto ar_comm = resource_manager_->get_ar_comm();
  ar_comm->all_reduce(ar_handle_, stream, device_id);
}

template <typename T>
GroupedExchangeWgrad<T>::GroupedExchangeWgrad(
    const std::shared_ptr<ResourceManager>& resource_manager)
    : resource_manager_(resource_manager), num_gpus_(resource_manager->get_local_gpu_count()) {
  bufs_.resize(num_gpus_, NULL);
  network_wgrad_buffs_.resize(num_gpus_, NULL);
  embed_wgrad_buffs_.resize(num_gpus_, NULL);

  for (size_t g = 0; g < num_gpus_; g++) {
    bufs_[g] = GeneralBuffer2<CudaAllocator>::create();
    network_wgrad_buffs_[g] = bufs_[g]->create_block<T>();
    embed_wgrad_buffs_[g] = bufs_[g]->create_block<T>();
  }

  auto ar_comm = resource_manager_->get_ar_comm();
  ar_handle_ = ar_comm->register_coll();
}

template <typename T>
void GroupedExchangeWgrad<T>::allocate() {
  int alignment = 16 * num_gpus_;
  for (size_t g = 0; g < num_gpus_; g++) {
    auto& gpu_resource = resource_manager_->get_local_gpu(g);
    CudaDeviceContext context(gpu_resource->get_device_id());
    bufs_[g]->allocate_aligned(alignment);
  }

  network_wgrad_size_ = network_wgrad_buffs_[0]->as_tensor().get_size_in_bytes();
  if (network_wgrad_size_ % alignment != 0) {
    network_wgrad_size_ += (alignment - (network_wgrad_size_ % alignment));
  }

  auto ar_comm = resource_manager_->get_ar_comm();
  for (size_t g = 0; g < num_gpus_; g++) {
    ar_comm->set_coll_buf(ar_handle_, bufs_[g]->get_ptr(), bufs_[g]->get_size_in_bytes(), g);
  }
  ar_comm->register_coll_buf(ar_handle_);
}

template <typename T>
void GroupedExchangeWgrad<T>::update_embed_wgrad_size(size_t size) {
  int alignment = 16 * num_gpus_;
  if (size % alignment != 0) {
    size += (alignment - (size % alignment));
  }
  embed_wgrad_size_ = size;

  auto ar_comm = resource_manager_->get_ar_comm();
  ar_comm->update_size(ar_handle_, network_wgrad_size_ + embed_wgrad_size_);
}

template <typename T>
void GroupedExchangeWgrad<T>::allreduce(size_t device_id, cudaStream_t stream) {
  auto ar_comm = resource_manager_->get_ar_comm();
  ar_comm->all_reduce(ar_handle_, stream, device_id);
}

template class NetworkExchangeWgrad<__half>;
template class NetworkExchangeWgrad<float>;
template class GroupedExchangeWgrad<__half>;
template class GroupedExchangeWgrad<float>;
}  // namespace HugeCTR

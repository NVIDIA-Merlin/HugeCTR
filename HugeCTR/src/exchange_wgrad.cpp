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

#include <exchange_wgrad.hpp>
#include <utils.hpp>

namespace HugeCTR {

template <typename T>
NetworkExchangeWgrad<T>::NetworkExchangeWgrad(
    const std::shared_ptr<ResourceManager>& resource_manager)
    : resource_manager_(resource_manager), num_gpus_(resource_manager->get_local_gpu_count()) {
  // TODO remove it after Hybrid embedding is deprecated
  null_wgrad_buffs_.resize(num_gpus_, nullptr);
  auto ar_comm = resource_manager_->get_ar_comm();
  ar_handle_ = ar_comm->register_coll();
}
template <typename T>
void NetworkExchangeWgrad<T>::init_ar_comm(const std::vector<void*>& ptr, size_t sizes) {
  network_wgrad_size_ = sizes;
  auto ar_comm = resource_manager_->get_ar_comm();
  for (size_t g = 0; g < num_gpus_; g++) {
    HCTR_CHECK_HINT(ptr[g], "buffer does not exist");
    ar_comm->set_coll_buf(ar_handle_, ptr[g], network_wgrad_size_, g);
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
  // TODO remove it after Hybrid embedding is deprecated
  embed_wgrad_buffs_.resize(num_gpus_, nullptr);

  auto ar_comm = resource_manager_->get_ar_comm();
  ar_handle_ = ar_comm->register_coll();
}
template <typename T>
void GroupedExchangeWgrad<T>::init_ar_comm(const std::vector<void*>& ptr, size_t sizes) {
  network_wgrad_size_ = sizes;
  auto ar_comm = resource_manager_->get_ar_comm();
  for (size_t g = 0; g < num_gpus_; g++) {
    HCTR_CHECK_HINT(ptr[g], "buffer does not exist");
    ar_comm->set_coll_buf(ar_handle_, ptr[g], network_wgrad_size_, g);
  }
  ar_comm->register_coll_buf(ar_handle_);
}

template <typename T>
void GroupedExchangeWgrad<T>::update_embed_wgrad_size(size_t size) {
  // do nothing
  return;
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

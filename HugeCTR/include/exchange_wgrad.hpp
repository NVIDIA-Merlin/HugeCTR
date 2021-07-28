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

#pragma once
#include <general_buffer2.hpp>
#include <resource_manager.hpp>

namespace HugeCTR {

template <typename T>
using BuffPtr = std::shared_ptr<BufferBlock2<T>>;

template <typename T>
using BuffPtrs = std::vector<BuffPtr<T>>;

class ExchangeWgrad {
 public:
  virtual void allocate() = 0;
  virtual void update_embed_wgrad_size(size_t size) = 0;
  virtual void allreduce(size_t device_id, cudaStream_t stream) = 0;
};

template <typename TypeFP>
class NetworkExchangeWgrad : public ExchangeWgrad {
 public:
  const BuffPtrs<TypeFP>& get_network_wgrad_buffs() const  { return network_wgrad_buffs_; }
  const BuffPtrs<TypeFP>& get_embed_wgrad_buffs() const  { return null_wgrad_buffs_; }
  void allocate() final;
  void update_embed_wgrad_size(size_t size) final;
  void allreduce(size_t device_id, cudaStream_t stream);
  NetworkExchangeWgrad(const std::shared_ptr<ResourceManager>& resource_manager);
  ~NetworkExchangeWgrad() = default;

 private:

  BuffPtrs<TypeFP> network_wgrad_buffs_;
  BuffPtrs<TypeFP> null_wgrad_buffs_;
  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> bufs_;
  std::shared_ptr<ResourceManager> resource_manager_;

  AllReduceInPlaceComm::Handle ar_handle_;

  size_t network_wgrad_size_ = 0;
  size_t num_gpus_ = 0;
};

template <typename TypeFP>
class GroupedExchangeWgrad : public ExchangeWgrad {
 public:
  const BuffPtrs<TypeFP>& get_network_wgrad_buffs() const  { return network_wgrad_buffs_; }
  const BuffPtrs<TypeFP>& get_embed_wgrad_buffs() const  { return embed_wgrad_buffs_; }
  void allocate() final;
  void update_embed_wgrad_size(size_t size) final;
  void allreduce(size_t device_id, cudaStream_t stream);
  GroupedExchangeWgrad(const std::shared_ptr<ResourceManager>& resource_manager);
  ~GroupedExchangeWgrad() = default;

 private:

  BuffPtrs<TypeFP> network_wgrad_buffs_;
  BuffPtrs<TypeFP> embed_wgrad_buffs_;
  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> bufs_;
  std::shared_ptr<ResourceManager> resource_manager_;

  AllReduceInPlaceComm::Handle ar_handle_;
  
  size_t network_wgrad_size_ = 0;
  size_t embed_wgrad_size_ = 0;
  size_t num_gpus_ = 0;
};
}  // namespace HugeCTR


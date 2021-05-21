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

#pragma once
#include <cpu_resource.hpp>
#include <device_map.hpp>
#include <gpu_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <collectives/ib_comm.hpp>
#include <collectives/all_reduce_comm.hpp>

namespace HugeCTR {

/**
 * @brief GPU resources container.
 *
 * A GPU resource container in one node. An instant includes:
 * GPU resource vector, thread pool for training, nccl communicators.
 */
class ResourceManager {
  int num_process_;
  int process_id_;
  DeviceMap device_map_;
  DeviceMap::Layout device_layout_;
  std::shared_ptr<CPUResource> cpu_resource_;
  std::vector<std::shared_ptr<GPUResource>> gpu_resources_; /**< GPU resource vector */
  std::vector<std::vector<bool>> p2p_matrix_;

  std::vector<std::shared_ptr<rmm::mr::device_memory_resource>> base_cuda_mr_;
  std::vector<std::shared_ptr<rmm::mr::device_memory_resource>> memory_resource_;

#ifdef ENABLE_MPI
  std::unique_ptr<IbComm> ib_comm_ = NULL;
#endif
  std::shared_ptr<AllReduceInPlaceComm> ar_comm_ = NULL;

  void enable_all_peer_accesses();

  void all2all_warmup();

 public:
  ResourceManager(int num_process, int process_id, DeviceMap&& device_map, unsigned long long seed);
  static std::shared_ptr<ResourceManager> create(
      const std::vector<std::vector<int>>& visible_devices, unsigned long long seed,
      DeviceMap::Layout dist = DeviceMap::LOCAL_FIRST);
  ResourceManager(const ResourceManager&) = delete;
  ResourceManager& operator=(const ResourceManager&) = delete;
  int get_num_process() const { return num_process_; }
  int get_process_id() const { return process_id_; }
  int get_master_process_id() const { return 0; }
  bool is_master_process() const { return process_id_ == 0; }

#ifdef ENABLE_MPI
  IbComm* get_ib_comm() const { return ib_comm_.get(); }
  void set_ready_to_transfer() { if (ib_comm_) ib_comm_->set_ready_to_transfer(); }
#endif
  void set_ar_comm(AllReduceAlgo algo, bool use_mixed_precision);
  AllReduceInPlaceComm* get_ar_comm() const { return ar_comm_.get(); }

  DeviceMap::Layout get_device_layout() const { return device_map_.get_device_layout(); }

  const std::shared_ptr<CPUResource>& get_local_cpu() const { return cpu_resource_; }

  const std::shared_ptr<GPUResource>& get_local_gpu(size_t local_gpu_id) const {
    return gpu_resources_[local_gpu_id];
  }
  const std::vector<int>& get_local_gpu_device_id_list() const {
    return device_map_.get_device_list();
  }
  size_t get_local_gpu_count() const { return device_map_.get_device_list().size(); }
  size_t get_global_gpu_count() const { return device_map_.size(); }

  int get_process_id_from_gpu_global_id(size_t global_gpu_id) const {
    return device_map_.get_pid(global_gpu_id);
  }

  size_t get_gpu_local_id_from_global_id(size_t global_gpu_id) const {  // sequential GPU indices
    return device_map_.get_local_id(global_gpu_id);
  }

  size_t get_gpu_global_id_from_local_id(size_t local_gpu_id) const {  // sequential GPU indices
    return device_map_.get_global_id(local_gpu_id);
  }

  bool p2p_enabled(int src_dev, int dst_dev) const;
  bool all_p2p_enabled() const;

  const std::shared_ptr<rmm::mr::device_memory_resource>& get_device_rmm_device_memory_resource(
      int local_gpu_id) const {
    auto dev_list = device_map_.get_device_list();
    auto it = std::find(dev_list.begin(), dev_list.end(), local_gpu_id);
    auto index = std::distance(dev_list.begin(), it);
    return memory_resource_[index];
  }
};
}  // namespace HugeCTR

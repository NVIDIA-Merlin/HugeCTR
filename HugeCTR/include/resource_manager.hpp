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
#include <cpu_resource.hpp>
#include <device_map.hpp>
#include <gpu_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace HugeCTR {

/**
 * @brief GPU resources container.
 *
 * A GPU resource container in one node. An instant includes:
 * GPU resource vector, thread pool for training, nccl communicators.
 */
class ResourceManager {
  int num_process_;
  int pid_;
  DeviceMap device_map_;
  std::shared_ptr<CPUResource> cpu_resource_;
  std::vector<std::shared_ptr<GPUResource>> gpu_resources_; /**< GPU resource vector */
  std::vector<std::vector<bool>> p2p_matrix_;

  std::shared_ptr<rmm::mr::device_memory_resource> memory_resource_;

  void enable_all_peer_accesses();
  ResourceManager(int num_process, int pid, DeviceMap&& device_map, unsigned long long seed);

 public:
  static std::shared_ptr<ResourceManager> create(
      const std::vector<std::vector<int>>& visible_devices, unsigned long long seed);
  ResourceManager(const ResourceManager&) = delete;
  ResourceManager& operator=(const ResourceManager&) = delete;

  int get_num_process() const { return num_process_; }
  int get_pid() const { return pid_; }

  const std::shared_ptr<CPUResource>& get_local_cpu() const { return cpu_resource_; }

  const std::shared_ptr<GPUResource>& get_local_gpu(size_t local_gpu_id) const {
    return gpu_resources_[local_gpu_id];
  }
  const std::vector<int>& get_local_gpu_device_id_list() const {
    return device_map_.get_device_list();
  }
  size_t get_local_gpu_count() const { return device_map_.get_device_list().size(); }
  size_t get_global_gpu_count() const { return device_map_.size(); }

  int get_pid_from_gpu_global_id(size_t global_gpu_id) const {
    return device_map_.get_pid(global_gpu_id);
  }

  size_t get_gpu_local_id_from_global_id(size_t global_gpu_id) const {  // sequential GPU indices
    return device_map_.get_local_id(global_gpu_id);
  }

  bool p2p_enabled(int src_dev, int dst_dev) const;
  bool all_p2p_enabled() const;

  const std::shared_ptr<rmm::mr::device_memory_resource>& get_rmm_mr_device_memory_resource()
      const {
    return memory_resource_;
  }
};
}  // namespace HugeCTR
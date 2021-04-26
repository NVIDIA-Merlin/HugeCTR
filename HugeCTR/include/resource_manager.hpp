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
#include <resource_manager_base.hpp>
#include <cpu_resource.hpp>
#include <device_map.hpp>
#include <gpu_resource.hpp>

namespace HugeCTR {

/**
 * @brief Second-level ResourceManager interface
 *
 * The second level resource manager interface shared by training and inference
 */
class ResourceManager : public ResourceManagerBase {
 public:
  static std::shared_ptr<ResourceManager> create(
      const std::vector<std::vector<int>>& visible_devices, unsigned long long seed);
  virtual int get_num_process() const = 0;
  virtual int get_process_id() const = 0;
  virtual int get_master_process_id() const = 0;
  virtual bool is_master_process() const = 0;
  virtual const std::shared_ptr<CPUResource>& get_local_cpu() const = 0;
  virtual const std::vector<int>& get_local_gpu_device_id_list() const = 0;
  virtual int get_process_id_from_gpu_global_id(size_t global_gpu_id) const = 0;
  virtual size_t get_gpu_local_id_from_global_id(size_t global_gpu_id) const = 0;
  virtual size_t get_gpu_global_id_from_local_id(size_t local_gpu_id) const = 0;
  virtual bool p2p_enabled(int src_dev, int dst_dev) const = 0;
  virtual bool all_p2p_enabled() const = 0;
};
}  // namespace HugeCTR

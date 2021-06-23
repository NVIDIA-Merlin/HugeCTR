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
#include <collectives/ib_comm.hpp>
#include <collectives/all_reduce_comm.hpp>

namespace HugeCTR {

/**
 * @brief Second-level ResourceManager interface
 *
 * The second level resource manager interface shared by training and inference
 */
class ResourceManager : public ResourceManagerBase {
#ifdef ENABLE_MPI
  std::unique_ptr<IbComm> ib_comm_ = NULL;
#endif
  std::shared_ptr<AllReduceInPlaceComm> ar_comm_ = NULL;

  void all2all_warmup();

 public:
  ResourceManager(int num_process, int process_id, DeviceMap&& device_map, unsigned long long seed);
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

#ifdef ENABLE_MPI
  IbComm* get_ib_comm() const { return ib_comm_.get(); }
  void set_ready_to_transfer() { if (ib_comm_) ib_comm_->set_ready_to_transfer(); }
#endif
  void set_ar_comm(AllReduceAlgo algo, bool use_mixed_precision);
  AllReduceInPlaceComm* get_ar_comm() const { return ar_comm_.get(); }

  DeviceMap::Layout get_device_layout() const { return device_map_.get_device_layout(); }
};
}  // namespace HugeCTR

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
#include <resource_manager.hpp>

namespace HugeCTR {

/**
 * @brief GPU resources manager which holds the minimal, essential set of resources
 *
 * A core GPU Resource manager
 */
class ResourceManagerCore : public ResourceManager {
 private:
  int num_process_;
  int process_id_;
  DeviceMap device_map_;
  std::shared_ptr<CPUResource> cpu_resource_;
  std::vector<std::shared_ptr<GPUResource>> gpu_resources_; /**< GPU resource vector */
  std::vector<std::vector<bool>> p2p_matrix_;

  void all2all_warmup();
  void enable_all_peer_accesses();

 public:
  ResourceManagerCore(int num_process, int process_id, DeviceMap&& device_map,
                      unsigned long long seed);
  ResourceManagerCore(const ResourceManagerCore&) = delete;
  ResourceManagerCore& operator=(const ResourceManagerCore&) = delete;

  // from ResourceManagerBase
  void set_local_gpu(std::shared_ptr<GPUResource> gpu_resource, size_t local_gpu_id) override {
    if (local_gpu_id >= get_local_gpu_count()) {
      CK_THROW_(Error_t::WrongInput, "Error: Invalid local_gpu_id");
    }
    if (gpu_resources_[local_gpu_id] != nullptr) {
      CK_THROW_(Error_t::WrongInput, "Error: Already initialized");
    }
    gpu_resources_[local_gpu_id] = gpu_resource;
  }
  const std::vector<std::shared_ptr<GPUResource>>& get_local_gpus() const override {
    return gpu_resources_;
  }
  const std::shared_ptr<GPUResource>& get_local_gpu(size_t local_gpu_id) const override {
    return gpu_resources_[local_gpu_id];
  }
  size_t get_local_gpu_count() const override { return device_map_.get_device_list().size(); }
  size_t get_global_gpu_count() const override { return device_map_.size(); }


  // from ResourceManager
  int get_num_process() const override { return num_process_; }
  int get_process_id() const override { return process_id_; }
  int get_master_process_id() const override { return 0; }
  bool is_master_process() const override { return process_id_ == 0; }

  const std::shared_ptr<CPUResource>& get_local_cpu() const override { return cpu_resource_; }

  const std::vector<int>& get_local_gpu_device_id_list() const override {
    return device_map_.get_device_list();
  }

  int get_process_id_from_gpu_global_id(size_t global_gpu_id) const override {
    return device_map_.get_pid(global_gpu_id);
  }

  size_t get_gpu_local_id_from_global_id(size_t global_gpu_id) const override {
    return device_map_.get_local_id(global_gpu_id);
  }

  size_t get_gpu_global_id_from_local_id(size_t local_gpu_id) const override {
    return device_map_.get_global_id(local_gpu_id);
  }

  bool p2p_enabled(int src_dev, int dst_dev) const override;
  bool all_p2p_enabled() const override;

  DeviceMap::Layout get_device_layout() const override { return device_map_.get_device_layout(); }

#ifdef ENABLE_MPI
  IbComm* get_ib_comm() const override { 
    CK_THROW_(Error_t::IllegalCall, "Error: should not be reached");
    return nullptr;
  }
  void set_ready_to_transfer() override { 
    CK_THROW_(Error_t::IllegalCall, "Error: should not be reached");
  }
#endif
  void set_ar_comm(AllReduceAlgo algo, bool use_mixed_precision) override {
    CK_THROW_(Error_t::IllegalCall, "Error: should not be reached");
  }
  AllReduceInPlaceComm* get_ar_comm() const override {
    CK_THROW_(Error_t::IllegalCall, "Error: should not be reached");
    return nullptr;
  }

};
}  // namespace HugeCTR

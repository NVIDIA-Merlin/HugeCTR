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

#include <resource_managers/resource_manager_ext.hpp>
#include <utils.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#pragma GCC diagnostic pop

namespace HugeCTR {

void ResourceManagerExt::initialize_rmm_resources() {
  const size_t pool_alloc_size = 256 * 1024 * 1024;

  using dmmr = rmm::mr::device_memory_resource;

  CudaDeviceContext context;
  auto local_gpu_device_id_list = core_->get_local_gpu_device_id_list();
  for (size_t i = 0; i < local_gpu_device_id_list.size(); i++) {
    context.set_device(local_gpu_device_id_list[i]);
    base_cuda_mr_.emplace_back(
        std::shared_ptr<rmm::mr::cuda_memory_resource>(new rmm::mr::cuda_memory_resource()));
    memory_resource_.emplace_back(std::shared_ptr<rmm::mr::pool_memory_resource<dmmr>>(
        new rmm::mr::pool_memory_resource<dmmr>(base_cuda_mr_.back().get(), pool_alloc_size)));
    rmm::mr::set_current_device_resource(memory_resource_.back().get());
  }
}

std::shared_ptr<ResourceManager> ResourceManagerExt::create(
    const std::vector<std::vector<int>>& visible_devices, unsigned long long seed) {
  auto core = ResourceManager::create(visible_devices, seed);

  return std::shared_ptr<ResourceManager>(new ResourceManagerExt(core));
}

ResourceManagerExt::ResourceManagerExt(std::shared_ptr<ResourceManager> core)
    : core_(core) {
  initialize_rmm_resources();
}

const std::shared_ptr<rmm::mr::device_memory_resource>&
ResourceManagerExt::get_device_rmm_device_memory_resource(int local_gpu_id) const {
  auto dev_list = core_->get_local_gpu_device_id_list();
  auto it = std::find(dev_list.begin(), dev_list.end(), local_gpu_id);
  auto index = std::distance(dev_list.begin(), it);
  return memory_resource_[index];
}

}  // namespace HugeCTR

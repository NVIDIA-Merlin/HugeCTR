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

#include <resource_manager.hpp>
#include <random>
#include <resource_managers/resource_manager_core.hpp>
#include <utils.hpp>

namespace HugeCTR {
// std::unordered_map<int, int> CudaCPUDeviceContext::device_id_to_numa_node_;

std::shared_ptr<ResourceManager> ResourceManager::create(
    const std::vector<std::vector<int>>& visible_devices, unsigned long long seed,
    DeviceMap::Layout layout) {
  int size = 1, rank = 0;

#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &size));
  CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
#endif

  DeviceMap device_map(visible_devices, rank, layout);

  std::random_device rd;
  if (seed == 0) {
    seed = rd();
  }

#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD));
#endif

  MESSAGE_("Global seed is " + std::to_string(seed));
  
  return std::shared_ptr<ResourceManager>(
      new ResourceManagerCore(size, rank, std::move(device_map), seed));
}

}  // namespace HugeCTR

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

#include <base/debug/logger.hpp>
#include <chrono>
#include <core23/allocator_factory.hpp>
#include <core23/allocator_params.hpp>
#include <core23/details/pool_cuda_allocator.hpp>
#include <cstdint>
#include <random>

namespace {

using namespace HugeCTR::core23;

const AllocatorParams g_allocator_params = {};

constexpr int64_t size_mb{1 << 20};

struct Allocation {
  void* ptr = nullptr;
  int64_t size = 0;
};

Allocation remove_at(std::vector<Allocation>& allocs, int64_t index) {
  auto removed = allocs[index];

  if ((allocs.size() > 1) && (index < allocs.size() - 1)) {
    std::swap(allocs[index], allocs.back());
  }
  allocs.pop_back();

  return removed;
}

void uniform_random_allocations(AllocatorParams allocator_params, Device device) {
  size_t free = 0, total = 0;
  HCTR_LIB_THROW(cudaMemGetInfo(&free, &total));
  HCTR_LOG_S(INFO, ROOT) << free / size_mb << " mb left in the beginning" << std::endl;

  int64_t num_allocations = 1000;
  int64_t max_allocation_size = 4096;
  size_t max_usage = free / size_mb;

  auto b_t = std::chrono::steady_clock::now();

  std::random_device r;
  std::default_random_engine e(r());

  constexpr int allocation_probability{100};
  constexpr int max_op_chance{99};

  std::uniform_int_distribution<int64_t> size_distribution(1, max_allocation_size * size_mb);
  std::uniform_int_distribution<int64_t> op_distribution(0, max_op_chance);
  std::uniform_int_distribution<int64_t> index_distribution(0, num_allocations - 1);

  max_usage *= size_mb;
  int64_t active_allocations = 0;
  int64_t allocation_count = 0;
  int64_t failed_allocations = 0;

  std::vector<Allocation> allocations{};
  int64_t allocation_size{0};

  auto allocator = GetAllocator(allocator_params, device);
  HCTR_LOG_S(INFO, ROOT) << free / size_mb << " mb left after creating an Allocator" << std::endl;
  CUDAStream stream = CUDAStream(cudaStreamNonBlocking, 0);

  for (int64_t i = 0; i < num_allocations; i++) {
    bool do_alloc = true;
    auto size = size_distribution(e);

    if (active_allocations > 0) {
      int64_t chance = op_distribution(e);
      do_alloc = (chance < allocation_probability);
    }

    void* ptr = nullptr;
    if (do_alloc) {
      try {
        ptr = allocator->allocate(size, stream);
      } catch (...) {
        do_alloc = false;
        failed_allocations++;
      }
    }

    if (do_alloc) {
      allocations.push_back({ptr, size});
      active_allocations++;
      allocation_count++;
      allocation_size += size;
    } else {
      if (active_allocations > 0) {
        int64_t index = index_distribution(e) % active_allocations;
        active_allocations--;
        Allocation to_free = remove_at(allocations, index);
        allocator->deallocate(to_free.ptr, stream);
        allocation_size -= to_free.size;
      }
    }
  }

  cudaDeviceSynchronize();
  auto e_t = std::chrono::steady_clock::now();

  HCTR_LOG_S(INFO, ROOT) << "Succeeded allocations: " << allocation_count
                         << ", Failed allocations: " << failed_allocations
                         << ", Active allocations: " << active_allocations
                         << ", Active allocation size: " << allocation_size / size_mb << " mb"
                         << std::endl;

  HCTR_LIB_THROW(cudaMemGetInfo(&free, &total));
  HCTR_LOG_S(INFO, ROOT) << free / size_mb << " mb left in the end" << std::endl;
  HCTR_LOG_S(INFO, ROOT) << std::chrono::duration_cast<std::chrono::milliseconds>(e_t - b_t).count()
                         << " ms" << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    bool pooled = true;
    if (argc >= 2) {
      std::istringstream(std::string(argv[1])) >> pooled;
    }

    AllocatorParams my_allocator_params = g_allocator_params;
    Device device(DeviceType::GPU, 0);
    if (pooled) {
      // TODO: change this line after introducing the ResourceManager
      my_allocator_params.custom_factory = [](const auto& params, const auto& device) {
        return std::unique_ptr<Allocator>(new PoolCUDAAllocator(device));
      };
    }
    uniform_random_allocations(my_allocator_params, device);
  } catch (...) {
    HCTR_LOG_S(INFO, ROOT) << "Something is wrong" << std::endl;
  }
}
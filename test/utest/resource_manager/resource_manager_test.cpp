#include <gtest/gtest.h>

#include <resource_managers/resource_manager_core.hpp>
using namespace HugeCTR;

void new_delete_resource_manager() {
  std::vector<int> device_list = {0};
  // auto gpu_id = device_list[0];
  std::vector<std::vector<int>> vvgpu;
  int numprocs = 1;
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto resource_manager = ResourceManagerCore::create(vvgpu, 424242);
}

TEST(RM, new_delete_resource_manager) {
  new_delete_resource_manager();
  new_delete_resource_manager();
}

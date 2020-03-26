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

#include "HugeCTR/include/device_map.hpp"
#include "gtest/gtest.h"

using namespace HugeCTR;

TEST(device_map, device_map_basic_teste) {
  std::vector<int> node1{0, 2, 3, 4}, node2{0, 1, 2}, node3{2, 3, 4, 5, 6};
  std::vector<std::vector<int>> nodes;
  nodes.push_back(node1);
  nodes.push_back(node2);
  nodes.push_back(node3);
  DeviceMap dm0(nodes, 0);
  DeviceMap dm1(nodes, 1);
  DeviceMap dm2(nodes, 2);
  std::cout << "dm0.size= " << dm0.size() << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << "global_id= " << i << "local_id= " << dm0.get_local_device_id(i) << std::endl;
    std::cout << "global_id= " << i << "local_id= " << dm0.get_pid(i) << std::endl;
    std::cout << "local_id= " << i << "global_id= " << dm0.get_global_id(i) << std::endl;
  }

  std::cout << "dm1.size= " << dm1.size() << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << "global_id= " << i << "local_id= " << dm1.get_local_device_id(i) << std::endl;
    std::cout << "global_id= " << i << "local_id= " << dm1.get_pid(i) << std::endl;
    std::cout << "local_id= " << i << "global_id= " << dm1.get_global_id(i) << std::endl;
  }

  std::cout << "dm2.size= " << dm2.size() << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << "global_id= " << i << "local_id= " << dm2.get_local_device_id(i) << std::endl;
    std::cout << "global_id= " << i << "local_id= " << dm2.get_pid(i) << std::endl;
    std::cout << "local_id= " << i << "global_id= " << dm2.get_global_id(i) << std::endl;
  }
}

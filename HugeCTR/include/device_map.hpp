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

#include <iostream>
#include <map>
#include <vector>
#include "HugeCTR/include/common.hpp"

namespace HugeCTR {

/**
 * @brief Device mapping.
 *
 * A class to map the devices between global id / local id and the actual GPU id.
 * @verbatim
 * For example we have two nodes and gpus: [[0,1,2,3],[1,2]]
 * global id: [0,1,2,3,4,5]
 * local id: [0,1,2,3,0,1]
 * device id: [0,1,2,3,1,2]
 * @endverbaim
 */
class DeviceMap {
 private:
  const std::vector<std::vector<int>> device_list_total_; /**< device list for all the nodes*/
  std::map<int, int> global_local_map_;                   /**< global to device id (gpu id) map */
  std::map<int, int> local_global_map_;    /**< device id (gpu id) to global id map */
  std::map<int, int> global_pid_map_;      /**< global id to process id map */
  std::map<int, int> global_local_id_map_; /**< global id to local id map */
  const int my_pid_;                       /**< process id for local node */
  const std::vector<int> device_list_;     /**< device list for local node */
 public:
  /**
   * Ctor.
   * Generate the maps.
   */
  DeviceMap(const std::vector<std::vector<int>>& device_list_total, int my_pid)
      : device_list_total_(device_list_total),
        my_pid_(my_pid),
        device_list_(device_list_total_[my_pid_]) {
    try {
      if (device_list_total_.size() <= (unsigned int)my_pid) {
        CK_THROW_(Error_t::WrongInput, "device_list_total_.size() <= my_pid");
      }
      int pid = 0;
      int global_id = 0;
      int local_id = 0;
      for (auto tmp_device_list : device_list_total_) {
        for (auto tmp_device : tmp_device_list) {
          if (pid == my_pid_) {
            global_local_map_.insert(std::pair<int, int>(global_id, tmp_device));
            local_global_map_.insert(std::pair<int, int>(tmp_device, global_id));
            global_local_id_map_.insert(std::pair<int, int>(global_id, local_id));
            local_id++;
          }
          global_pid_map_.insert(std::pair<int, int>(global_id, pid));
          global_id++;
        }
        pid++;
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
    }
  }

  const std::vector<int>& get_device_list() const { return device_list_; }

  /**
   * Get global id according to device id.
   * @param local_device_id device id.
   * @return global id, if cannot find this local device in current node return -1.
   */
  int get_global_id(int local_device_id) const {
    std::map<int, int>::const_iterator it = local_global_map_.find(local_device_id);
    if (it == local_global_map_.end()) {
      return -1;
    } else {
      return it->second;
    }
  }

  /**
   * Get device id according to global id.
   * @param global_id global id.
   * @return device_id, if cannot find this device in current node return -1.
   */
  int get_local_id(int global_id) const {
    std::map<int, int>::const_iterator it = global_local_id_map_.find(global_id);
    if (it == global_local_id_map_.end()) {
      return -1;
    } else {
      return it->second;
    }
  }

  /**
   * Get the total number of devices.
   */
  size_t size() const { return global_pid_map_.size(); }

  /**
   * Get the total number of nodes.
   */
  int num_nodes() const { return device_list_total_.size(); }

  /**
   * Get local id from global id, if not find return -1.
   */
  int get_local_device_id(int global_id) const {
    std::map<int, int>::const_iterator it = global_local_map_.find(global_id);
    if (it == global_local_map_.end()) {
      return -1;
    } else {
      return it->second;
    }
  }

  /**
   * Get process id from global id, if not find return -1.
   */
  int get_pid(int global_id) const {
    std::map<int, int>::const_iterator it = global_pid_map_.find(global_id);
    if (it == global_pid_map_.end()) {
      return -1;
    } else {
      return it->second;
    }
  }

  /**
   * Dtor.
   */
  ~DeviceMap() {}
};

}  // namespace HugeCTR

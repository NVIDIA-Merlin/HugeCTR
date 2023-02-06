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
#pragma once

#include <algorithm>
#include <common.hpp>
#include <map>
#include <vector>

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
 public:
  enum Layout { LOCAL_FIRST, NODE_FIRST };

 private:
  const std::vector<std::vector<int>> device_list_total_; /**< device list for all the nodes*/
  std::map<int, int> global_local_map_;                   /**< global to device id (gpu id) map */
  std::map<int, int> global_pid_map_;                     /**< global id to process id map */
  std::map<int, int> global_local_id_map_;                /**< global id to local id map */
  std::map<int, int> local_global_id_map_;                /**< local id to global id map */
  const int my_pid_;                                      /**< process id for local node */
  const int num_procs_;
  const std::vector<int> device_list_; /**< device list for local node */
  Layout device_layout_;

 public:
  /**
   * Ctor.
   * Generate the maps.
   */
  DeviceMap(const std::vector<std::vector<int>>& device_list_total, int my_pid,
            Layout layout = LOCAL_FIRST)
      : device_list_total_(device_list_total),
        my_pid_(my_pid),
        num_procs_(device_list_total.size()),
        device_list_(device_list_total_[my_pid_]),
        device_layout_(layout) {
    try {
      if (device_list_total_.size() <= (unsigned int)my_pid) {
        HCTR_OWN_THROW(Error_t::WrongInput, "device_list_total_.size() <= my_pid");
      }

      for (const std::vector<int>& device_list : device_list_total) {
        std::vector<int> tmp_device_list(device_list);
        auto it = std::unique(tmp_device_list.begin(), tmp_device_list.end());
        if (it != tmp_device_list.end()) {
          HCTR_OWN_THROW(Error_t::WrongInput, "duplicated device id");
        }
      }

      if (layout == LOCAL_FIRST) {
        int pid = 0;
        int global_id = 0;
        int local_id = 0;
        for (auto tmp_device_list : device_list_total_) {
          for (auto tmp_device : tmp_device_list) {
            if (pid == my_pid_) {
              global_local_map_.insert(std::pair<int, int>(global_id, tmp_device));
              global_local_id_map_.insert(std::pair<int, int>(global_id, local_id));
              local_global_id_map_.insert(std::pair<int, int>(local_id, global_id));
              local_id++;
            }
            global_pid_map_.insert(std::pair<int, int>(global_id, pid));
            global_id++;
          }
          pid++;
        }
      } else if (layout == NODE_FIRST) {
        // Need to have same number of devices on all nodes else A2A won't work
        HCTR_LOG(INFO, ROOT, "Using NODE_FIRST layout\n");
        unsigned int mysize = device_list_.size();
        for (auto tmp_device_list : device_list_total_) {
          if (tmp_device_list.size() != mysize) {
            HCTR_OWN_THROW(Error_t::WrongInput,
                           "All nodes should have same number of devices for NODE_FIRST layout");
          }
        }

        int pid = 0;
        for (auto& tmp_device_list : device_list_total_) {
          int local_id = 0;
          for (auto& tmp_device : tmp_device_list) {
            int global_id = local_id * num_procs_ + pid;
            if (pid == my_pid_) {
              global_local_map_.insert(std::pair<int, int>(global_id, tmp_device));
              global_local_id_map_.insert(std::pair<int, int>(global_id, local_id));
              local_global_id_map_.insert(std::pair<int, int>(local_id, global_id));
            }
            global_pid_map_.insert(std::pair<int, int>(global_id, pid));
            local_id++;
          }
          pid++;
        }
      } else {
        throw std::runtime_error("[HCDEBUG][ERROR] Runtime error: Invalid device layout");
      }
    } catch (const std::runtime_error& rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    }
  }

  const std::vector<int>& get_device_list() const { return device_list_; }

  /**
   * Get global id according to local id.
   * @param local_id local id.
   * @return global id, if cannot find this local device in current node return -1.
   */
  int get_global_id(int local_id) const {
    std::map<int, int>::const_iterator it = local_global_id_map_.find(local_id);
    if (it == local_global_id_map_.end()) {
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

  Layout get_device_layout() const { return device_layout_; }

  /**
   * Dtor.
   */
  ~DeviceMap() {}
};

}  // namespace HugeCTR

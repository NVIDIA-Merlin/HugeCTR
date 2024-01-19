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

#include <collectives/all_reduce_comm.hpp>
#include <collectives/ib_comm.hpp>
#include <resource_manager.hpp>

namespace HugeCTR {

/**
 * @brief GPU resources manager which holds all the resources required by training
 *
 * An extended GPU Resource manager
 */
class CollectiveManager {
  std::shared_ptr<ResourceManager> core_;

#ifdef ENABLE_MPI
  std::unique_ptr<IbComm> ib_comm_ = NULL;
#endif
  std::shared_ptr<AllReduceInPlaceComm> ar_comm_ = NULL;

 public:
  CollectiveManager() = default;
  CollectiveManager(const std::shared_ptr<ResourceManager>& core) : core_(core) {}

  HCTR_DISALLOW_COPY_AND_MOVE(CollectiveManager);

#ifdef ENABLE_MPI
  void init_ib_comm();
  IbComm* get_ib_comm() const { return ib_comm_.get(); }
  void set_ready_to_transfer() {
    if (ib_comm_) ib_comm_->set_ready_to_transfer();
  }
#endif
  void set_ar_comm(AllReduceAlgo algo, bool use_mixed_precision);
  AllReduceInPlaceComm* get_ar_comm() const { return ar_comm_.get(); }
};
}  // namespace HugeCTR

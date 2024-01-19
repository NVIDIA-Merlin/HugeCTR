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

#include <collectives/collective.hpp>
#include <core23/logger.hpp>

namespace HugeCTR {

#ifdef ENABLE_MPI
void CollectiveManager::init_ib_comm() {
  int num_process = core_->get_num_process();
  if (num_process > 1) {
    int process_id = core_->get_process_id();
    ib_comm_ = std::make_unique<IbComm>();
    ib_comm_->init(num_process, core_->get_local_gpu_count(), process_id,
                   core_->get_local_gpu_device_id_list());
  }
}
#endif

void CollectiveManager::set_ar_comm(AllReduceAlgo algo, bool use_mixed_precision) {
  int num_process = core_->get_num_process();
#ifdef ENABLE_MPI
  IbComm* ib_comm_ptr = nullptr;
  if (algo == AllReduceAlgo::ONESHOT) {
    init_ib_comm();
    ib_comm_ptr = ib_comm_.get();
  }
  ar_comm_ = AllReduceInPlaceComm::create(num_process, algo, use_mixed_precision,
                                          core_->get_local_gpus(), ib_comm_ptr);
#else
  ar_comm_ =
      AllReduceInPlaceComm::create(num_process, algo, use_mixed_precision, core_->get_local_gpus());
#endif
}

}  // namespace HugeCTR

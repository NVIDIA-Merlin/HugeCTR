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
#include <memory>
#include "HugeCTR/include/common.hpp"

namespace HugeCTR {
class GPUBarrier {
 public:
  GPUBarrier(size_t num_gpus, const std::vector<int>& device_list,
             bool enforce_order = false);
  ~GPUBarrier();

  void sync_all_gpus(const cudaStream_t* streams);
  void sync_all_gpus(const cudaStream_t stream, size_t device_id);
  void sync_all_gpus_global(const cudaStream_t stream, size_t device_id);
  void sync_all_gpus_report_host(size_t** d_report_count, size_t* h_report_ptr,
                                 const cudaStream_t* streams);
  void sync_all_gpus_report_host(size_t* d_report_count, size_t* h_report_ptr,
                                 const cudaStream_t stream, size_t device_id);
  void sync_all_gpus_report_host_and_inc(size_t* d_report_count, size_t* h_report_ptr,
                                         const cudaStream_t stream, size_t device_id);

 private:
  size_t** d_barrier_flags_ = NULL;
  float** d_global_barrier_store_ = NULL;
  size_t*** d_rem_barrier_flags_ = NULL;
  size_t num_gpus_;
  std::vector<int> dev_list_;
  bool enforce_order_;
};
}  // namespace HugeCTR

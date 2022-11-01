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

#include <gpu_barrier.hpp>
#include <utils.cuh>

namespace HugeCTR {
namespace gpu_barrier {
__device__ __forceinline__ void sync_all_gpus_func(size_t** d_rem_barrier_flags, size_t my_local_id,
                                                   size_t ndevs) {
  size_t count = d_rem_barrier_flags[my_local_id][my_local_id];
  size_t g_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (g_tid < ndevs) {
    volatile size_t* rem_flag = d_rem_barrier_flags[g_tid];
    volatile size_t* my_flag = d_rem_barrier_flags[my_local_id];
    rem_flag[my_local_id] = (count + 1);
    while (my_flag[g_tid] < (count + 1)) {
    }
  }
  __syncthreads();
}

__global__ void sync_all_gpus_cuda(size_t** d_rem_barrier_flags, size_t my_local_id, size_t ndevs,
                                   bool enforce_order = false) {
  if (enforce_order) {
    __threadfence_system();
  }
  sync_all_gpus_func(d_rem_barrier_flags, my_local_id, ndevs);
}

// Only single CTA launch
__global__ void sync_all_gpus_report_host_cuda(size_t** d_rem_barrier_flags, size_t* d_report_count,
                                               size_t* h_report_ptr, size_t my_local_id,
                                               size_t ndevs) {
  size_t g_tid = blockIdx.x * blockDim.x + threadIdx.x;
  sync_all_gpus_func(d_rem_barrier_flags, my_local_id, ndevs);
  if ((g_tid == 0) && (my_local_id == 0)) {
    *h_report_ptr = *d_report_count;
  }
}

__global__ void sync_all_gpus_report_host_and_inc_cuda(size_t** d_rem_barrier_flags,
                                                       size_t* d_report_count, size_t* h_report_ptr,
                                                       size_t my_local_id, size_t ndevs,
                                                       bool enforce_order = false) {
  if (enforce_order) {
    __threadfence_system();
  }
  size_t g_tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t count = *d_report_count;
  sync_all_gpus_func(d_rem_barrier_flags, my_local_id, ndevs);
  if ((g_tid == 0) && (my_local_id == 0)) {
    *h_report_ptr = count;
  }
  if (g_tid == 0) {
    *d_report_count = (count + 1);
  }
  __syncthreads();
}
}  // namespace gpu_barrier

using namespace gpu_barrier;

GPUBarrier::GPUBarrier(size_t num_gpus, const std::vector<int>& dev_list, bool enforce_order)
    : num_gpus_(num_gpus), dev_list_(dev_list), enforce_order_(enforce_order) {
  d_barrier_flags_ = new size_t*[num_gpus_];
  d_rem_barrier_flags_ = new size_t**[num_gpus_];
  d_global_barrier_store_ = new float*[num_gpus_];

  for (size_t g = 0; g < num_gpus_; g++) {
    HCTR_LIB_THROW(cudaSetDevice(dev_list_[g]));
    HCTR_LIB_THROW(cudaMalloc(&d_barrier_flags_[g], num_gpus_ * sizeof(size_t)));
    HCTR_LIB_THROW(cudaMemset(d_barrier_flags_[g], 0, num_gpus_ * sizeof(size_t)));
  }

  for (size_t g = 0; g < num_gpus_; g++) {
    HCTR_LIB_THROW(cudaSetDevice(dev_list_[g]));
    HCTR_LIB_THROW(cudaMalloc(&d_rem_barrier_flags_[g], num_gpus_ * sizeof(size_t*)));
    HCTR_LIB_THROW(cudaMemcpy(d_rem_barrier_flags_[g], d_barrier_flags_,
                              num_gpus_ * sizeof(size_t*), cudaMemcpyHostToDevice));
  }

  for (size_t g = 0; g < num_gpus_; g++) {
    HCTR_LIB_THROW(cudaSetDevice(dev_list_[g]));
    HCTR_LIB_THROW(cudaMalloc(&d_global_barrier_store_[g], sizeof(float)));
  }
}

void GPUBarrier::sync_all_gpus(const cudaStream_t* streams) {
  constexpr size_t MAX_TPB = 256;
  size_t n_blocks = ceildiv<size_t>(num_gpus_, MAX_TPB);
  for (size_t g = 0; g < num_gpus_; g++) {
    HCTR_LIB_THROW(cudaSetDevice(dev_list_[g]));
    sync_all_gpus_cuda<<<n_blocks, MAX_TPB, 0, streams[g]>>>(d_rem_barrier_flags_[g], g, num_gpus_);
  }
}

void GPUBarrier::sync_all_gpus(const cudaStream_t stream, size_t device_id) {
  constexpr size_t MAX_TPB = 256;
  size_t n_blocks = ceildiv<size_t>(num_gpus_, MAX_TPB);
  HCTR_LIB_THROW(cudaSetDevice(dev_list_[device_id]));
  sync_all_gpus_cuda<<<n_blocks, MAX_TPB, 0, stream>>>(d_rem_barrier_flags_[device_id], device_id,
                                                       num_gpus_, enforce_order_);
}

void GPUBarrier::sync_all_gpus_report_host(size_t** d_report_count, size_t* h_report_ptr,
                                           const cudaStream_t* streams) {
  for (size_t g = 0; g < num_gpus_; g++) {
    HCTR_LIB_THROW(cudaSetDevice(dev_list_[g]));
    sync_all_gpus_report_host_cuda<<<1, num_gpus_, 0, streams[g]>>>(
        d_rem_barrier_flags_[g], d_report_count[g], h_report_ptr, g, num_gpus_);
  }
}

void GPUBarrier::sync_all_gpus_report_host(size_t* d_report_count, size_t* h_report_ptr,
                                           const cudaStream_t stream, size_t device_id) {
  HCTR_LIB_THROW(cudaSetDevice(dev_list_[device_id]));
  sync_all_gpus_report_host_cuda<<<1, num_gpus_, 0, stream>>>(
      d_rem_barrier_flags_[device_id], d_report_count, h_report_ptr, device_id, num_gpus_);
}

void GPUBarrier::sync_all_gpus_report_host_and_inc(size_t* d_report_count, size_t* h_report_ptr,
                                                   const cudaStream_t stream, size_t device_id) {
  HCTR_LIB_THROW(cudaSetDevice(dev_list_[device_id]));
  sync_all_gpus_report_host_and_inc_cuda<<<1, num_gpus_, 0, stream>>>(
      d_rem_barrier_flags_[device_id], d_report_count, h_report_ptr, device_id, num_gpus_,
      enforce_order_);
}

GPUBarrier::~GPUBarrier() {
  for (size_t g = 0; g < num_gpus_; g++) {
    HCTR_LIB_THROW(cudaSetDevice(dev_list_[g]));
    HCTR_LIB_THROW(cudaFree(d_rem_barrier_flags_[g]));
    HCTR_LIB_THROW(cudaFree(d_barrier_flags_[g]));
    HCTR_LIB_THROW(cudaFree(d_global_barrier_store_[g]));
  }
  delete[] d_rem_barrier_flags_;
  delete[] d_barrier_flags_;
  delete[] d_global_barrier_store_;
}
}  // namespace HugeCTR

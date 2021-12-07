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

#include <common.hpp>
#include <gpu_resource.hpp>
#include <utils.hpp>

namespace HugeCTR {

GPUResource::GPUResource(int device_id, size_t local_id, size_t global_id,
                         unsigned long long replica_uniform_seed,
                         unsigned long long replica_variant_seed, const ncclComm_t& comm)
    : device_id_(device_id), local_id_(local_id), global_id_(global_id), comm_(comm) {
  CudaDeviceContext context(device_id);
  HCTR_LIB_THROW(cudaStreamCreateWithFlags(&computation_stream_, cudaStreamNonBlocking));
  HCTR_LIB_THROW(cudaStreamCreateWithFlags(&memcpy_stream_, cudaStreamNonBlocking));
  HCTR_LIB_THROW(cudaStreamCreateWithFlags(&computation_stream_2_, cudaStreamNonBlocking));
  HCTR_LIB_THROW(cudaStreamCreateWithFlags(&p2p_stream_, cudaStreamNonBlocking));
  HCTR_LIB_THROW(cudaEventCreate(&compute_sync_event_));
  HCTR_LIB_THROW(cudaEventCreate(&compute2_sync_event_));
  HCTR_LIB_THROW(cudaEventCreate(&wait_wgrad_event_));
  HCTR_LIB_THROW(
      curandCreateGenerator(&replica_uniform_curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  HCTR_LIB_THROW(
      curandSetPseudoRandomGeneratorSeed(replica_uniform_curand_generator_, replica_uniform_seed));
  HCTR_LIB_THROW(curandSetStream(replica_uniform_curand_generator_, computation_stream_));
  HCTR_LIB_THROW(
      curandCreateGenerator(&replica_variant_curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  HCTR_LIB_THROW(
      curandSetPseudoRandomGeneratorSeed(replica_variant_curand_generator_, replica_variant_seed));
  HCTR_LIB_THROW(curandSetStream(replica_variant_curand_generator_, computation_stream_));
  HCTR_LIB_THROW(cublasCreate(&cublas_handle_));
  HCTR_LIB_THROW(cublasCreate(&cublas_handle_wgrad_));
  HCTR_LIB_THROW(cublasLtCreate(&cublaslt_handle_));
  HCTR_LIB_THROW(cublasSetStream(cublas_handle_, computation_stream_));
  HCTR_LIB_THROW(cublasSetStream(cublas_handle_wgrad_, computation_stream_2_));
  HCTR_LIB_THROW(cudnnCreate(&cudnn_handle_));
  HCTR_LIB_THROW(cudnnSetStream(cudnn_handle_, computation_stream_));

  int sm_count;
  HCTR_LIB_THROW(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  sm_count_ = sm_count;

  HCTR_LIB_THROW(cudaDeviceGetAttribute(&cc_major_, cudaDevAttrComputeCapabilityMajor, device_id));
  HCTR_LIB_THROW(cudaDeviceGetAttribute(&cc_minor_, cudaDevAttrComputeCapabilityMinor, device_id));
}

GPUResource::~GPUResource() {
  try {
    CudaDeviceContext context(device_id_);
    HCTR_LIB_THROW(ncclCommDestroy(comm_));
    HCTR_LIB_THROW(curandDestroyGenerator(replica_uniform_curand_generator_));
    HCTR_LIB_THROW(curandDestroyGenerator(replica_variant_curand_generator_));
    HCTR_LIB_THROW(cublasDestroy(cublas_handle_));
    HCTR_LIB_THROW(cublasDestroy(cublas_handle_wgrad_));
    HCTR_LIB_THROW(cudnnDestroy(cudnn_handle_));
    HCTR_LIB_THROW(cudaEventDestroy(compute_sync_event_));
    HCTR_LIB_THROW(cudaEventDestroy(compute2_sync_event_));
    HCTR_LIB_THROW(cudaEventDestroy(wait_wgrad_event_));
    HCTR_LIB_THROW(cudaStreamDestroy(computation_stream_));
    HCTR_LIB_THROW(cudaStreamDestroy(memcpy_stream_));
    HCTR_LIB_THROW(cudaStreamDestroy(computation_stream_2_));
    HCTR_LIB_THROW(cudaStreamDestroy(p2p_stream_));
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

void GPUResource::set_compute_event_sync(const cudaStream_t& sync_stream) {
  HCTR_LIB_THROW(cudaEventRecord(compute_sync_event_, sync_stream));
  return;
}

void GPUResource::wait_on_compute_event(const cudaStream_t& sync_stream) {
  HCTR_LIB_THROW(cudaStreamWaitEvent(sync_stream, compute_sync_event_));
  return;
}

void GPUResource::set_compute2_event_sync(const cudaStream_t& sync_stream) {
  HCTR_LIB_THROW(cudaEventRecord(compute2_sync_event_, sync_stream));
  return;
}

void GPUResource::wait_on_compute2_event(const cudaStream_t& sync_stream) {
  HCTR_LIB_THROW(cudaStreamWaitEvent(sync_stream, compute2_sync_event_));
  return;
}

void GPUResource::set_wgrad_event_sync(const cudaStream_t& sync_stream) const {
  CK_CUDA_THROW_(cudaEventRecord(wait_wgrad_event_, sync_stream));
  return;
}

void GPUResource::wait_on_wgrad_event(const cudaStream_t& sync_stream) const {
  CK_CUDA_THROW_(cudaStreamWaitEvent(sync_stream, wait_wgrad_event_));
  return;
}

}  // namespace HugeCTR

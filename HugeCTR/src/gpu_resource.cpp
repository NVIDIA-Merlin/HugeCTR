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

GPUResource::GPUResource(int device_id, size_t local_id, size_t global_id, unsigned long long replica_uniform_seed,
                         unsigned long long replica_variant_seed, const ncclComm_t& comm)
    : device_id_(device_id), local_id_(local_id), global_id_(global_id), comm_(comm) {
  CudaDeviceContext context(device_id);
  CK_CUDA_THROW_(cudaStreamCreateWithFlags(&computation_stream_, cudaStreamNonBlocking));
  CK_CUDA_THROW_(cudaStreamCreateWithFlags(&memcpy_stream_, cudaStreamNonBlocking));
  CK_CUDA_THROW_(cudaStreamCreateWithFlags(&computation_stream_2_, cudaStreamNonBlocking));
  CK_CUDA_THROW_(cudaEventCreate(&compute_sync_event_));
  CK_CUDA_THROW_(cudaEventCreate(&compute2_sync_event_));
  CK_CURAND_THROW_(
      curandCreateGenerator(&replica_uniform_curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CK_CURAND_THROW_(
      curandSetPseudoRandomGeneratorSeed(replica_uniform_curand_generator_, replica_uniform_seed));
  CK_CURAND_THROW_(curandSetStream(replica_uniform_curand_generator_, computation_stream_));
  CK_CURAND_THROW_(
      curandCreateGenerator(&replica_variant_curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CK_CURAND_THROW_(
      curandSetPseudoRandomGeneratorSeed(replica_variant_curand_generator_, replica_variant_seed));
  CK_CURAND_THROW_(curandSetStream(replica_variant_curand_generator_, computation_stream_));
  CK_CUBLAS_THROW_(cublasCreate(&cublas_handle_));
  CK_CUBLAS_THROW_(cublasLtCreate(&cublaslt_handle_));
  CK_CUBLAS_THROW_(cublasSetStream(cublas_handle_, computation_stream_));
  CK_CUDNN_THROW_(cudnnCreate(&cudnn_handle_));
  CK_CUDNN_THROW_(cudnnSetStream(cudnn_handle_, computation_stream_));

  int sm_count;
  CK_CUDA_THROW_(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  sm_count_ = sm_count;

  CK_CUDA_THROW_(cudaDeviceGetAttribute(&cc_major_, cudaDevAttrComputeCapabilityMajor, device_id));
  CK_CUDA_THROW_(cudaDeviceGetAttribute(&cc_minor_, cudaDevAttrComputeCapabilityMinor, device_id));
}

GPUResource::~GPUResource() {
  try {
    CudaDeviceContext context(device_id_);
    CK_NCCL_THROW_(ncclCommDestroy(comm_));
    CK_CURAND_THROW_(curandDestroyGenerator(replica_uniform_curand_generator_));
    CK_CURAND_THROW_(curandDestroyGenerator(replica_variant_curand_generator_));
    CK_CUBLAS_THROW_(cublasDestroy(cublas_handle_));
    CK_CUDNN_THROW_(cudnnDestroy(cudnn_handle_));
    CK_CUDA_THROW_(cudaEventDestroy(compute_sync_event_));
    CK_CUDA_THROW_(cudaEventDestroy(compute2_sync_event_));
    CK_CUDA_THROW_(cudaStreamDestroy(computation_stream_));
    CK_CUDA_THROW_(cudaStreamDestroy(memcpy_stream_));
    CK_CUDA_THROW_(cudaStreamDestroy(computation_stream_2_));
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

void GPUResource::set_compute_event_sync(const cudaStream_t& sync_stream) {
  CK_CUDA_THROW_(cudaEventRecord(compute_sync_event_, sync_stream));
  return;
}

void GPUResource::wait_on_compute_event(const cudaStream_t& sync_stream) {
  CK_CUDA_THROW_(cudaStreamWaitEvent(sync_stream, compute_sync_event_));
  return;
}

void GPUResource::set_compute2_event_sync(const cudaStream_t& sync_stream) {
  CK_CUDA_THROW_(cudaEventRecord(compute2_sync_event_, sync_stream));
  return;
}

void GPUResource::wait_on_compute2_event(const cudaStream_t& sync_stream) {
  CK_CUDA_THROW_(cudaStreamWaitEvent(sync_stream, compute2_sync_event_));
  return;
}

}  // namespace HugeCTR

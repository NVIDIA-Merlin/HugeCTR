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

#include <common.hpp>
#include <gpu_resource.hpp>
#include <utils.hpp>

namespace HugeCTR {

GPUResource::GPUResource(int device_id, size_t local_id, size_t global_id,
                         unsigned long long replica_uniform_seed,
                         unsigned long long replica_variant_seed, const ncclComm_t& comm)
    : device_id_(device_id),
      local_id_(local_id),
      global_id_(global_id),
      stream_name_("default"),
      comm_(comm) {
  CudaDeviceContext context(device_id);
  HCTR_LIB_THROW(
      curandCreateGenerator(&replica_variant_curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  HCTR_LIB_THROW(
      curandSetPseudoRandomGeneratorSeed(replica_variant_curand_generator_, replica_variant_seed));
  HCTR_LIB_THROW(cublasCreate(&cublas_handle_));
  HCTR_LIB_THROW(cudnnCreate(&cudnn_handle_));

  set_stream(stream_name_, 0);
  cudaStream_t computation_stream_ = stream_event_manager_.get_stream(stream_name_);
  memcpy_stream_ = stream_event_manager_.get_stream("memcpy_stream_", cudaStreamNonBlocking);
  computation_stream_2_ = stream_event_manager_.get_stream("computation_stream_2_");
  p2p_stream_ = stream_event_manager_.get_stream("p2p_stream_", cudaStreamNonBlocking);
  wait_wgrad_event_ = stream_event_manager_.get_event("wgrad_event", cudaEventDefault);

  HCTR_LIB_THROW(
      curandCreateGenerator(&replica_uniform_curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  HCTR_LIB_THROW(
      curandSetPseudoRandomGeneratorSeed(replica_uniform_curand_generator_, replica_uniform_seed));
  HCTR_LIB_THROW(curandSetStream(replica_uniform_curand_generator_, computation_stream_));

  HCTR_LIB_THROW(curandSetStream(replica_variant_curand_generator_, computation_stream_));
  HCTR_LIB_THROW(cublasCreate(&cublas_handle_wgrad_));
  HCTR_LIB_THROW(cublasLtCreate(&cublaslt_handle_));
  HCTR_LIB_THROW(cublasSetStream(cublas_handle_, computation_stream_));
  HCTR_LIB_THROW(cublasSetStream(cublas_handle_wgrad_, computation_stream_2_));
  HCTR_LIB_THROW(cudnnSetStream(cudnn_handle_, computation_stream_));

  int sm_count;
  HCTR_LIB_THROW(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  sm_count_ = sm_count;
  int max_thread_per_sm;
  HCTR_LIB_THROW(cudaDeviceGetAttribute(&max_thread_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor,
                                        device_id));
  max_thread_per_sm_ = max_thread_per_sm;

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
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
}

void GPUResource::set_wgrad_event_sync(const cudaStream_t& sync_stream) const {
  HCTR_LIB_THROW(cudaEventRecord(wait_wgrad_event_, sync_stream));
  return;
}

void GPUResource::wait_on_wgrad_event(const cudaStream_t& sync_stream) const {
  HCTR_LIB_THROW(cudaStreamWaitEvent(sync_stream, wait_wgrad_event_));
  return;
}

const cudaStream_t& GPUResource::get_stream(const std::string& name, int priority) {
  return stream_event_manager_.get_stream(name, cudaStreamNonBlocking, priority);
}

const cudaEvent_t& GPUResource::get_event(const std::string& name) {
  return stream_event_manager_.get_event(name, cudaEventDisableTiming);
}

}  // namespace HugeCTR

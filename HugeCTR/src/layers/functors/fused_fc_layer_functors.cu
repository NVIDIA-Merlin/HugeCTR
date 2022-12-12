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
#include <layers/functors/fused_fc_layer_functors.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

template <typename T>
void CublasFusedFCLayerDesc<T>::set_fprop_attr(const T* bias_ptr, Activation_t act, T* mask_out_ptr,
                                               size_t batch_size, size_t bottom_size,
                                               size_t top_size, bool enable_tf32_compute) {
  fprop_desc.set_fprop_attr({batch_size, bottom_size}, {bottom_size, top_size}, CUBLAS_OP_N,
                            CUBLAS_OP_N, CUBLASLT_ORDER_ROW, enable_tf32_compute, bias_ptr, act,
                            mask_out_ptr);
}

template <typename T>
void CublasFusedFCLayerDesc<T>::set_bprop_attr(T* dbias_bottom_ptr, T* dbias_top_ptr,
                                               T* mask_in_ptr, size_t batch_size,
                                               size_t bottom_size, size_t top_size,
                                               bool enable_tf32_compute) {
  bprop_wgrad_desc.set_bprop_attr({batch_size, bottom_size}, {batch_size, top_size}, CUBLAS_OP_T,
                                  CUBLAS_OP_N, CUBLASLT_ORDER_ROW, enable_tf32_compute,
                                  dbias_top_ptr, nullptr);
  bprop_dgrad_desc.set_bprop_attr({batch_size, top_size}, {bottom_size, top_size}, CUBLAS_OP_N,
                                  CUBLAS_OP_T, CUBLASLT_ORDER_ROW, enable_tf32_compute,
                                  dbias_bottom_ptr, mask_in_ptr);
}

template <typename T>
void CublasFusedFCLayerAlgo<T>::set_fprop_algo(const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                                               cublasLtHandle_t cublaslt_handle) {
  fprop_algo.init_algorithm(cublas_layer_desc.fprop_desc, cublaslt_handle);
}

template <typename T>
void CublasFusedFCLayerAlgo<T>::set_bprop_algo(const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                                               cublasLtHandle_t cublaslt_handle) {
  bprop_wgrad_algo.init_algorithm(cublas_layer_desc.bprop_wgrad_desc, cublaslt_handle);
  bprop_dgrad_algo.init_algorithm(cublas_layer_desc.bprop_dgrad_desc, cublaslt_handle);
}

template <typename T>
void CublasFusedFCLayerAlgo<T>::search_algorithm(T* bottom, T* top, T* kernel, size_t batch_size,
                                                 size_t input_size, size_t output_size,
                                                 const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                                                 cublasLtHandle_t cublaslt_handle,
                                                 cudaStream_t stream) {
  fprop_algo.search_algorithm(1.0f, bottom, kernel, 0.0f, top, top, cublas_layer_desc.fprop_desc,
                              cublaslt_handle, stream);
  bprop_wgrad_algo.search_algorithm(1.0f, bottom, top, 1.0f, kernel, kernel,
                                    cublas_layer_desc.bprop_wgrad_desc, cublaslt_handle, stream);
  bprop_dgrad_algo.search_algorithm(1.0f, top, kernel, 0.0f, bottom, bottom,
                                    cublas_layer_desc.bprop_dgrad_desc, cublaslt_handle, stream);
}

template <typename T>
void FusedFCLayerFunctors<T>::fprop(const T* kernel, const T* bottom, T* top,
                                    const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                                    const CublasFusedFCLayerAlgo<T>& cublas_layer_algo,
                                    cublasLtHandle_t cublaslt_handle, cudaStream_t stream) {
  gemm_functor_(1.0f, bottom, kernel, 0.0f, top, top, cublas_layer_desc.fprop_desc,
                cublas_layer_algo.fprop_algo, cublaslt_handle, stream);
}

namespace {
__global__ void reverse_relu_kernel(__half* dRelu, const __half* mask, const __half* dY, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n / 2) return;
  const size_t num_threads = blockDim.x * gridDim.x;
  const __half2 zero = TypeFunc<__half2>::zero();
  __half2* dRelu2 = reinterpret_cast<__half2*>(dRelu);
  const __half2* mask2 = reinterpret_cast<const __half2*>(mask);
  const __half2* dY2 = reinterpret_cast<const __half2*>(dY);
  __half2 m = __hgt2(mask2[tid], zero);
  dRelu2[tid] = __hmul2(__ldg(dY2 + tid), m);
  if (tid + num_threads >= n / 2) return;
  m = __hgt2(mask2[tid + num_threads], zero);
  dRelu2[tid + num_threads] = __hmul2(__ldg(dY2 + tid + num_threads), m);
}
__global__ void reverse_relu_kernel(float* dRelu, const float* mask, const float* dY, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n / 2) return;
  float2* dRelu2 = reinterpret_cast<float2*>(dRelu);
  const float2* mask2 = reinterpret_cast<const float2*>(mask);
  const float2* dY2 = reinterpret_cast<const float2*>(dY);

  const float2 m = __ldg(mask2 + tid);
  const float2 d = __ldg(dY2 + tid);
  dRelu2[tid] = {m.x > 0.0f ? d.x : 0.0f, m.y > 0.0f ? d.y : 0.0f};
}
__global__ void reverse_relu_kernel_not_aligned(__half* dRelu, const __half* mask, const __half* dY,
                                                size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const __half zero = TypeFunc<__half>::zero();
  if (tid >= n) return;
  __half m = __hgt(mask[tid], zero);
  dRelu[tid] = __hmul(__ldg(dY + tid), m);
}
__global__ void reverse_relu_kernel_not_aligned(float* dRelu, const float* mask, const float* dY,
                                                size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  dRelu[tid] = __ldg(dY + tid) * (mask[tid] > 0);
}
}  // namespace

template <typename T>
void FusedFCLayerFunctors<T>::bprop(const T* kernel, const T* bottom, const T* train_top,
                                    const T* mask_aux, size_t mask_aux_size, T* grad_top,
                                    T* bottom_bprop, T* kernel_grad,
                                    const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                                    const CublasFusedFCLayerAlgo<T>& cublas_layer_algo,
                                    cublasLtHandle_t cublaslt_handle, cudaStream_t stream,
                                    cudaStream_t overlap_stream, cudaEvent_t& event_overlap,
                                    bool async_wgrad, bool skip_dgrad) {
  if (mask_aux != nullptr) {
    if constexpr (std::is_same<T, float>::value) {
      if (mask_aux_size % 2 == 0) {
        reverse_relu_kernel<<<(mask_aux_size / 2 - 1) / 1024 + 1, 1024, 0, stream>>>(
            grad_top, mask_aux, train_top, mask_aux_size);
      } else {
        reverse_relu_kernel_not_aligned<<<(mask_aux_size - 1) / 1024 + 1, 1024, 0, stream>>>(
            grad_top, mask_aux, train_top, mask_aux_size);
      }
    } else {
      if (mask_aux_size % 4 == 0) {
        reverse_relu_kernel<<<(mask_aux_size / 4 - 1) / 1024 + 1, 1024, 0, stream>>>(
            grad_top, mask_aux, train_top, mask_aux_size);
      } else {
        reverse_relu_kernel_not_aligned<<<(mask_aux_size - 1) / 1024 + 1, 1024, 0, stream>>>(
            grad_top, mask_aux, train_top, mask_aux_size);
      }
    }
  }

  // wait for dact
  if (async_wgrad) {
    HCTR_LIB_THROW(cudaEventRecord(event_overlap, stream));
    HCTR_LIB_THROW(cudaStreamWaitEvent(overlap_stream, event_overlap));
  }

  gemm_functor_(1.0f, bottom, grad_top, 1.0f, kernel_grad, kernel_grad,
                cublas_layer_desc.bprop_wgrad_desc, cublas_layer_algo.bprop_wgrad_algo,
                cublaslt_handle, async_wgrad ? overlap_stream : stream);

  if (!skip_dgrad) {
    gemm_functor_(1.0f, grad_top, kernel, 0.0f, bottom_bprop, bottom_bprop,
                  cublas_layer_desc.bprop_dgrad_desc, cublas_layer_algo.bprop_dgrad_algo,
                  cublaslt_handle, stream);
  }
}

template <typename T>
void FusedFCLayerFunctors<T>::search_algorithm(T* bottom, T* top, T* kernel, size_t batch_size,
                                               size_t input_size, size_t output_size,
                                               const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                                               CublasFusedFCLayerAlgo<T>& cublas_layer_algo,
                                               cublasLtHandle_t cublaslt_handle,
                                               cudaStream_t stream) {
  cublas_layer_algo.search_algorithm(bottom, top, kernel, batch_size, input_size, output_size,
                                     cublas_layer_desc, cublaslt_handle, stream);
}

}  // namespace HugeCTR

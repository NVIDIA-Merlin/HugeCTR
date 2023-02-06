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

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <common.hpp>
#include <layer.hpp>
#include <layers/functors/fused_gemm_functors.hpp>
#include <type_traits>

namespace HugeCTR {

template <typename T>
struct CublasFusedFCLayerDesc {
  CublasDesc<T> fprop_desc;
  CublasDesc<T> bprop_wgrad_desc;
  CublasDesc<T> bprop_dgrad_desc;

  void set_fprop_attr(const T* bias_ptr, Activation_t act, T* mask_out_ptr, size_t batch_size,
                      size_t bottom_size, size_t top_size, bool enable_tf32_compute);
  void set_bprop_attr(T* dbias_bottom_ptr, T* dbias_top_ptr, T* mask_in_ptr, size_t batch_size,
                      size_t bottom_size, size_t top_size, bool enable_tf32_compute);
};

template <typename T>
struct CublasFusedFCLayerAlgo {
  CublasAlgo<T> fprop_algo;
  CublasAlgo<T> bprop_wgrad_algo;
  CublasAlgo<T> bprop_dgrad_algo;

  void set_fprop_algo(const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                      cublasLtHandle_t cublaslt_handle);
  void set_bprop_algo(const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                      cublasLtHandle_t cublaslt_handle);
  void search_algorithm(T* bottom, T* top, T* kernel, size_t batch_size, size_t input_size,
                        size_t output_size, const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                        cublasLtHandle_t cublaslt_handle, cudaStream_t stream);
};

template <typename T>
class FusedFCLayerFunctors {
  GemmFunctor<T> gemm_functor_;

 public:
  void fprop(const T* kernel, const T* bottom, T* top,
             const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
             const CublasFusedFCLayerAlgo<T>& cublas_layer_algo, cublasLtHandle_t cublaslt_handle,
             cudaStream_t stream);

  void bprop(const T* kernel, const T* bottom, const T* train_top, const T* mask_aux,
             size_t mask_aux_size, T* grad_top, T* bottom_bprop, T* kernel_grad,
             const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
             const CublasFusedFCLayerAlgo<T>& cublas_layer_algo, cublasLtHandle_t cublaslt_handle,
             cudaStream_t stream, cudaStream_t overlap_stream, cudaEvent_t& event_overlap,
             bool async_wgrad, bool skip_dgrad);

  void search_algorithm(T* bottom, T* top, T* kernel, size_t batch_size, size_t input_size,
                        size_t output_size, const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                        CublasFusedFCLayerAlgo<T>& cublas_layer_algo,
                        cublasLtHandle_t cublaslt_handle, cudaStream_t stream);
};

template class CublasFusedFCLayerDesc<float>;
template class CublasFusedFCLayerDesc<__half>;

template class CublasFusedFCLayerAlgo<float>;
template class CublasFusedFCLayerAlgo<__half>;

template class FusedFCLayerFunctors<float>;
template class FusedFCLayerFunctors<__half>;

}  // namespace HugeCTR
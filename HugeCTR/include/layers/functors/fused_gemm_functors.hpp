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

namespace HugeCTR {

template <typename T>
struct CublasDesc {
  cublasLtMatmulDesc_t cublas_op_desc = NULL;
  cublasLtMatrixLayout_t cublas_mat_a_desc = NULL;
  cublasLtMatrixLayout_t cublas_mat_b_desc = NULL;
  cublasLtMatrixLayout_t cublas_mat_c_desc = NULL;
  cublasLtEpilogue_t epilogue;
  bool row_major = true;

  void set_fprop_attr(std::vector<size_t> dims_a, std::vector<size_t> dims_b,
                      cublasOperation_t op_a, cublasOperation_t op_b, cublasLtOrder_t order,
                      bool enable_tf32_compute, const T* bias_ptr = nullptr,
                      Activation_t act = Activation_t::None, T* mask_out_ptr = nullptr);
  void set_bprop_attr(std::vector<size_t> dims_a, std::vector<size_t> dims_b,
                      cublasOperation_t op_a, cublasOperation_t op_b, cublasLtOrder_t order,
                      bool enable_tf32_compute, T* dbias_ptr = nullptr,
                      const T* mask_in_ptr = nullptr);

  ~CublasDesc();
};

template <typename T>
struct CublasAlgo {
  size_t cublaslt_workspace_size = 1024 * 1024 * 8;
  void* cublaslt_workspace;
  cublasLtMatmulAlgo_t algo;
  cublasLtMatmulPreference_t cublas_preference = NULL;
  bool initialized = false;

  void init_algorithm(const CublasDesc<T>& cublas_desc, cublasLtHandle_t cublaslt_handle);

  void search_algorithm(const float alpha, const T* mat_a, const T* mat_b, const float beta,
                        const T* mat_c, T* mat_d, const CublasDesc<T>& cublas_desc,
                        cublasLtHandle_t cublaslt_handle, cudaStream_t stream);
  ~CublasAlgo();
};

template <typename T>
struct GemmFunctor {
  // D = alpha*(A*B) + beta*(C),
  void operator()(const float alpha, const T* mat_a, const T* mat_b, const float beta,
                  const T* mat_c, T* mat_d, const CublasDesc<T>& cublas_desc,
                  const CublasAlgo<T>& cublas_algo, cublasLtHandle_t cublaslt_handle,
                  cudaStream_t stream);
};

template class CublasDesc<float>;
template class CublasDesc<__half>;

template class CublasAlgo<float>;
template class CublasAlgo<__half>;

template class GemmFunctor<float>;
template class GemmFunctor<__half>;
}  // namespace HugeCTR
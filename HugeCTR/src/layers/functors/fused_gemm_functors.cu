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

#include <layers/functors/fused_gemm_functors.hpp>

namespace HugeCTR {
template <typename T>
void CublasDesc<T>::set_fprop_attr(std::vector<size_t> dims_a, std::vector<size_t> dims_b,
                                   cublasOperation_t op_a, cublasOperation_t op_b,
                                   cublasLtOrder_t order, bool enable_tf32_compute,
                                   const T* bias_ptr, Activation_t act, T* mask_out_ptr) {
  if (order == CUBLASLT_ORDER_ROW) {
    row_major = true;
    std::swap(dims_a, dims_b);
    std::swap(op_a, op_b);
    std::reverse(dims_a.begin(), dims_a.end());
    std::reverse(dims_b.begin(), dims_b.end());
    // CUBLASLT_ORDER_ROW cannot be combined with CUBLASLT_EPILOGUE_BIAS, so this workaround is
    // needed. It treats the row-major matrix as a transpose of the col-major matrix.
  } else {
    row_major = false;
  }

  cublasComputeType_t compute_type =
      enable_tf32_compute ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  HCTR_LIB_THROW(cublasLtMatmulDescCreate(&cublas_op_desc, compute_type, CUDA_R_32F));

  HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a,
                                                sizeof(op_a)));
  HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b,
                                                sizeof(op_b)));

  size_t cublas_rows_c = op_a == CUBLAS_OP_N ? dims_a[0] : dims_a[1];
  size_t cublas_cols_c = op_b == CUBLAS_OP_N ? dims_b[1] : dims_b[0];

  epilogue = mask_out_ptr != nullptr ? CUBLASLT_EPILOGUE_RELU_AUX : CUBLASLT_EPILOGUE_RELU;
  if (bias_ptr != nullptr) {
    epilogue = cublasLtEpilogue_t((int)epilogue | (int)CUBLASLT_EPILOGUE_BIAS);
  }
  if (act == Activation_t::None) {
    epilogue = bias_ptr == nullptr ? CUBLASLT_EPILOGUE_DEFAULT : CUBLASLT_EPILOGUE_BIAS;
  }

  HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));

  // Bias vector elements are the same type as alpha and beta when matrix D datatype is CUDA_R_8I
  // and same as matrix D datatype otherwise.
  if (bias_ptr != nullptr) {
    HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                  &bias_ptr, sizeof(bias_ptr)));
  }
  if (act != Activation_t::None) {
    HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc,
                                                  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                  &mask_out_ptr, sizeof(mask_out_ptr)));
    // relu_mask_ld must be divisible by 128 and be no less than the number of rows in the output
    // matrix.
    size_t relu_mask_ld = ((cublas_rows_c - 1) / 128 + 1) * 128;
    HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(
        cublas_op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &relu_mask_ld, sizeof(relu_mask_ld)));
  }

  uint32_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                &pointer_mode, sizeof(pointer_mode)));

  cudaDataType data_type = CUDA_R_32F;
  if constexpr (std::is_same<T, __half>::value) {
    data_type = CUDA_R_16F;
  }
  HCTR_LIB_THROW(
      cublasLtMatrixLayoutCreate(&cublas_mat_a_desc, data_type, dims_a[0], dims_a[1], dims_a[0]));
  HCTR_LIB_THROW(
      cublasLtMatrixLayoutCreate(&cublas_mat_b_desc, data_type, dims_b[0], dims_b[1], dims_b[0]));
  HCTR_LIB_THROW(cublasLtMatrixLayoutCreate(&cublas_mat_c_desc, data_type, cublas_rows_c,
                                            cublas_cols_c, cublas_rows_c));
}

template <typename T>
void CublasDesc<T>::set_bprop_attr(std::vector<size_t> dims_a, std::vector<size_t> dims_b,
                                   cublasOperation_t op_a, cublasOperation_t op_b,
                                   cublasLtOrder_t order, bool enable_tf32_compute, T* dbias_ptr,
                                   const T* mask_in_ptr) {
  if (order == CUBLASLT_ORDER_ROW) {
    row_major = true;
    std::swap(dims_a, dims_b);
    std::swap(op_a, op_b);
    std::reverse(dims_a.begin(), dims_a.end());
    std::reverse(dims_b.begin(), dims_b.end());
    // CUBLASLT_ORDER_ROW cannot be combined with CUBLASLT_EPILOGUE_BIAS, so this workaround is
    // needed. It treats the row-major matrix as a transpose of the col-major matrix.
  } else {
    row_major = false;
  }

  cublasComputeType_t compute_type =
      enable_tf32_compute ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  HCTR_LIB_THROW(cublasLtMatmulDescCreate(&cublas_op_desc, compute_type, CUDA_R_32F));

  HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a,
                                                sizeof(op_a)));
  HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b,
                                                sizeof(op_b)));
  size_t cublas_rows_c = op_a == CUBLAS_OP_N ? dims_a[0] : dims_a[1];
  size_t cublas_cols_c = op_b == CUBLAS_OP_N ? dims_b[1] : dims_b[0];

  if (mask_in_ptr != nullptr) {
    bool use_bgrad_bprop = dbias_ptr != nullptr;
    epilogue = use_bgrad_bprop ? CUBLASLT_EPILOGUE_DRELU_BGRAD : CUBLASLT_EPILOGUE_DRELU;
    HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epilogue, sizeof(epilogue)));
    if (use_bgrad_bprop) {
      HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(
          cublas_op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias_ptr, sizeof(dbias_ptr)));
    }
    HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc,
                                                  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                  &mask_in_ptr, sizeof(mask_in_ptr)));
    size_t relu_mask_ld = ((cublas_rows_c - 1) / 128 + 1) * 128;
    HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(
        cublas_op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &relu_mask_ld, sizeof(relu_mask_ld)));
  } else {
    bool use_bgrad_fuse_a = dbias_ptr != nullptr;
    epilogue = use_bgrad_fuse_a ? CUBLASLT_EPILOGUE_BGRADA : CUBLASLT_EPILOGUE_DEFAULT;
    HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epilogue, sizeof(epilogue)));
    if (use_bgrad_fuse_a) {
      HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(
          cublas_op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias_ptr, sizeof(dbias_ptr)));
    }
  }

  uint32_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  HCTR_LIB_THROW(cublasLtMatmulDescSetAttribute(cublas_op_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                &pointer_mode, sizeof(pointer_mode)));

  cudaDataType data_type = CUDA_R_32F;
  if constexpr (std::is_same<T, __half>::value) {
    data_type = CUDA_R_16F;
  }

  HCTR_LIB_THROW(
      cublasLtMatrixLayoutCreate(&cublas_mat_a_desc, data_type, dims_a[0], dims_a[1], dims_a[0]));
  HCTR_LIB_THROW(
      cublasLtMatrixLayoutCreate(&cublas_mat_b_desc, data_type, dims_b[0], dims_b[1], dims_b[0]));

  HCTR_LIB_THROW(cublasLtMatrixLayoutCreate(&cublas_mat_c_desc, data_type, cublas_rows_c,
                                            cublas_cols_c, cublas_rows_c));
}

template <typename T>
CublasDesc<T>::~CublasDesc() {
  cublasLtMatmulDescDestroy(cublas_op_desc);
  cublasLtMatrixLayoutDestroy(cublas_mat_a_desc);
  cublasLtMatrixLayoutDestroy(cublas_mat_b_desc);
  cublasLtMatrixLayoutDestroy(cublas_mat_c_desc);
}

template <typename T>
void CublasAlgo<T>::init_algorithm(const CublasDesc<T>& cublas_desc,
                                   cublasLtHandle_t cublaslt_handle) {
  HCTR_LIB_THROW(cublasLtMatmulPreferenceCreate(&cublas_preference));

  HCTR_LIB_THROW(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));
  HCTR_LIB_THROW(cublasLtMatmulPreferenceSetAttribute(
      cublas_preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size,
      sizeof(cublaslt_workspace_size)));

#if CUBLAS_VERSION < 120000
  uint32_t pointer_mode = CUBLASLT_POINTER_MODE_MASK_HOST;
  HCTR_LIB_THROW(cublasLtMatmulPreferenceSetAttribute(cublas_preference,
                                                      CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK,
                                                      &pointer_mode, sizeof(pointer_mode)));
  HCTR_LIB_THROW(
      cublasLtMatmulPreferenceSetAttribute(cublas_preference, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK,
                                           &cublas_desc.epilogue, sizeof(cublas_desc.epilogue)));
#endif

  cublasLtMatmulHeuristicResult_t heuristic_result;
  int returned_res = 0;
  HCTR_LIB_THROW(cublasLtMatmulAlgoGetHeuristic(
      cublaslt_handle, cublas_desc.cublas_op_desc, cublas_desc.cublas_mat_a_desc,
      cublas_desc.cublas_mat_b_desc, cublas_desc.cublas_mat_c_desc, cublas_desc.cublas_mat_c_desc,
      cublas_preference, 1, &heuristic_result, &returned_res));

  algo = heuristic_result.algo;

  if (returned_res == 0) {
    HCTR_LIB_THROW(CUBLAS_STATUS_NOT_SUPPORTED);
  }
  initialized = true;
}

template <typename T>
void CublasAlgo<T>::search_algorithm(const float alpha, const T* mat_a, const T* mat_b,
                                     const float beta, const T* mat_c, T* mat_d,
                                     const CublasDesc<T>& cublas_desc,
                                     cublasLtHandle_t cublaslt_handle, cudaStream_t stream) {
  if (cublas_desc.row_major) {
    std::swap(mat_a, mat_b);
  }
  if (!initialized) {
    init_algorithm(cublas_desc, cublaslt_handle);
  }
  const size_t repeat_num = 100;
  const int max_algo_count = 16;

  float shortestTime = std::numeric_limits<float>::max();
  float time;
  cudaEvent_t start, stop;
  HCTR_LIB_THROW(cudaEventCreate(&start));
  HCTR_LIB_THROW(cudaEventCreate(&stop));

  cublasLtMatmulHeuristicResult_t heuristic_result[max_algo_count] = {0};
  int algo_count = 0;
  HCTR_LIB_THROW(cublasLtMatmulAlgoGetHeuristic(
      cublaslt_handle, cublas_desc.cublas_op_desc, cublas_desc.cublas_mat_a_desc,
      cublas_desc.cublas_mat_b_desc, cublas_desc.cublas_mat_c_desc, cublas_desc.cublas_mat_c_desc,
      cublas_preference, max_algo_count, heuristic_result, &algo_count));

  if (algo_count == 0) {
    HCTR_LIB_THROW(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  for (int algoIdx = 0; algoIdx < algo_count; algoIdx++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    HCTR_LIB_THROW(cudaEventRecord(start, stream));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasLtMatmul(cublaslt_handle, cublas_desc.cublas_op_desc, &alpha, mat_a,
                              cublas_desc.cublas_mat_a_desc, mat_b, cublas_desc.cublas_mat_b_desc,
                              &beta, mat_c, cublas_desc.cublas_mat_c_desc, mat_d,
                              cublas_desc.cublas_mat_c_desc, &heuristic_result[algoIdx].algo,
                              cublaslt_workspace, cublaslt_workspace_size, stream);
    }
    HCTR_LIB_THROW(cudaEventRecord(stop, stream));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));

    time = time / repeat_num;
    if (status != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    if (time < shortestTime) {
      shortestTime = time;
      algo = heuristic_result[algoIdx].algo;
    }
  }

  HCTR_LIB_THROW(cudaEventDestroy(start));
  HCTR_LIB_THROW(cudaEventDestroy(stop));
}

template <typename T>
CublasAlgo<T>::~CublasAlgo() {
  cudaFree(cublaslt_workspace);
  cublasLtMatmulPreferenceDestroy(cublas_preference);
}

template <typename T>
void GemmFunctor<T>::operator()(const float alpha, const T* mat_a, const T* mat_b, const float beta,
                                const T* mat_c, T* mat_d, const CublasDesc<T>& cublas_desc,
                                const CublasAlgo<T>& cublas_algo, cublasLtHandle_t cublaslt_handle,
                                cudaStream_t stream) {
  if (cublas_desc.row_major) {
    std::swap(mat_a, mat_b);
  }
  HCTR_LIB_THROW(cublasLtMatmul(
      cublaslt_handle, cublas_desc.cublas_op_desc, &alpha, mat_a, cublas_desc.cublas_mat_a_desc,
      mat_b, cublas_desc.cublas_mat_b_desc, &beta, mat_c, cublas_desc.cublas_mat_c_desc, mat_d,
      cublas_desc.cublas_mat_c_desc, &cublas_algo.algo, cublas_algo.cublaslt_workspace,
      cublas_algo.cublaslt_workspace_size, stream));
}

}  // namespace HugeCTR

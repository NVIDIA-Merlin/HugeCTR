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

#include <algorithm>
#include <functional>
#include <layers/multi_head_attention_layer.hpp>
#include <utils.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

template <typename T>
MultiHeadAttentionLayer<T>::MultiHeadAttentionLayer(
    const Tensors2<T>& in_tensors, Tensor2<T>& out_tensor,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision,
    bool enable_tf32_compute)
    : Layer(gpu_resource),
      use_mixed_precision_(use_mixed_precision),
      enable_tf32_compute_(enable_tf32_compute) {
  try {
    num_ = in_tensors.size();

    // error input checking
    dims_ = in_tensors[0].get_dimensions().size();
    if (dims_ != 4) {
      HCTR_OWN_THROW(Error_t::WrongInput, "MultiHeadAttentionLayer needs 4D input tensors");
    }
    if (num_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "MultiHeadAttentionLayer needs 2 input tensors: query and key");
    }
    if (in_tensors[1].get_dimensions().size() != dims_) {
      HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
    }
    if (in_tensors[1].get_dimensions()[dims_ - 1] != in_tensors[0].get_dimensions()[dims_ - 1]) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "The last two dimension of the input tensors should be m x n, k x n");
    }
    if (in_tensors[0].get_dimensions()[0] != in_tensors[1].get_dimensions()[0]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "4D input tensors must have the same head number");
    }
    if (in_tensors[0].get_dimensions()[1] != in_tensors[1].get_dimensions()[1]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "4D input tensors must have the same batch size");
    }

    for (size_t i = 0; i < num_; i++) {
      in_tensors_.push_back(in_tensors[i]);
    }

    size_t m = in_tensors[0].get_dimensions()[dims_ - 2];
    size_t k = in_tensors[1].get_dimensions()[dims_ - 2];

    size_t h = in_tensors[0].get_dimensions()[0];
    size_t b = in_tensors[0].get_dimensions()[1];
    std::vector<size_t> out_dim = {h, b, m, k};
    blobs_buff->reserve(out_dim, &out_tensor);

    out_tensor_ = out_tensor;

    blobs_buff->reserve(in_tensors[0].get_dimensions(), &fprop_inputA_);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void MultiHeadAttentionLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  T* query = in_tensors_[0].get_ptr();
  T* key = in_tensors_[1].get_ptr();
  T* out = out_tensor_.get_ptr();

  const auto& in_tensor_dim = in_tensors_[0].get_dimensions();
  const auto& out_tensor_dim = out_tensor_.get_dimensions();

  size_t head_num, batch_size, from_seq_len, to_seq_len, size_per_head;

  head_num = in_tensor_dim[0];
  batch_size = in_tensor_dim[1];
  from_seq_len = in_tensor_dim[dims_ - 2];
  size_per_head = in_tensor_dim[dims_ - 1];
  to_seq_len = out_tensor_dim[dims_ - 1];

  const int batch_count = head_num * batch_size;
  const int m = from_seq_len;
  const int n = to_seq_len;
  const int k = size_per_head;
  float alpha = 1.0f, beta = 0.0f;
  long long int stride_a = static_cast<long long int>(to_seq_len) * size_per_head;
  long long int stride_b = static_cast<long long int>(from_seq_len) * size_per_head;
  long long int stride_c = static_cast<long long int>(from_seq_len) * to_seq_len;
  cudaDataType_t a_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;
  cudaDataType_t b_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;
  cudaDataType_t c_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;

  cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  cublasGemmAlgo_t algo =
      use_mixed_precision_ ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;

  HCTR_LIB_THROW(cublasGemmStridedBatchedEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                            n, m, k, &alpha, key, a_type, k, stride_a, query,
                                            b_type, k, stride_b, &beta, out, c_type, n, stride_c,
                                            batch_count, compute_type, algo));

  HCTR_LIB_THROW(cudaMemcpyAsync((void*)fprop_inputA_.get_ptr(), (void*)query,
                                 in_tensors_[0].get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void MultiHeadAttentionLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  T* query = in_tensors_[0].get_ptr();
  T* key = in_tensors_[1].get_ptr();
  T* out = out_tensor_.get_ptr();

  const auto& in_tensor_dim = in_tensors_[0].get_dimensions();
  const auto& out_tensor_dim = out_tensor_.get_dimensions();

  size_t head_num, batch_size, from_seq_len, to_seq_len, size_per_head;

  head_num = in_tensor_dim[0];
  batch_size = in_tensor_dim[1];
  from_seq_len = in_tensor_dim[dims_ - 2];
  size_per_head = in_tensor_dim[dims_ - 1];
  to_seq_len = out_tensor_dim[dims_ - 1];

  const int batch_count = head_num * batch_size;
  const int m = from_seq_len;
  const int n = size_per_head;
  const int k = to_seq_len;

  float alpha = 1.0f, beta = 0.0f;

  long long int stride_a = static_cast<long long int>(n) * k;
  long long int stride_b = static_cast<long long int>(k) * m;
  long long int stride_c = static_cast<long long int>(n) * m;
  cudaDataType_t a_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;
  cudaDataType_t b_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;
  cudaDataType_t c_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;

  cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  cublasGemmAlgo_t algo =
      use_mixed_precision_ ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;

  // gradient respect to query
  HCTR_LIB_THROW(cublasGemmStridedBatchedEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                            n, m, k, &alpha, key, a_type, n, stride_a, out, b_type,
                                            k, stride_b, &beta, query, c_type, n, stride_c,
                                            batch_count, compute_type, algo));
  T* cur_Q = fprop_inputA_.get_ptr();

  stride_a = static_cast<long long int>(n) * m;
  stride_b = static_cast<long long int>(m) * k;
  stride_c = static_cast<long long int>(n) * k;
  // gradient respect to key
  HCTR_LIB_THROW(cublasGemmStridedBatchedEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                                            n, k, m, &alpha, cur_Q, a_type, n, stride_a, out,
                                            b_type, k, stride_b, &beta, key, c_type, n, stride_c,
                                            batch_count, compute_type, algo));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class MultiHeadAttentionLayer<float>;
template class MultiHeadAttentionLayer<__half>;

}  // namespace HugeCTR

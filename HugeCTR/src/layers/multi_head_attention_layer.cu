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

// transpose from [batch_size, seq_len, hidden_num] -> [batch_size, head_num, seq_len,
// size_per_head]
template <typename T>
__global__ void transpose_QK(T* q_buf, T* k_buf, const T* Q, const T* K, const int batch_size,
                             const int seq_len, const int head_num, const int hidden_dim) {
  int d0_stride = hidden_dim * seq_len;
  int d1_stride = hidden_dim;
  int d2_stride = hidden_dim / head_num;

  int d0_out_stride = d0_stride;
  int d1_out_stride = d2_stride;
  int d2_out_stride = d2_stride * seq_len;

  int d0 = blockIdx.x;   // Batch
  int d1 = blockIdx.y;   // Sequence ID (0-127)
  int d2 = threadIdx.y;  // Head (0-11)
  int d3 = threadIdx.x;  // Values (groups of 4)

  float input_Q = Q[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
  q_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = input_Q;

  float input_K = K[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
  k_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = input_K;
}
// transpose from [batch_size, head_num, seq_len, size_per_head] -> [batch_size, seq_len,
// hidden_num]
template <typename T>
__global__ void transpose_QK_back(T* q_buf, T* k_buf, const T* Q, const T* K, const int batch_size,
                                  const int seq_len, const int head_num, const int hidden_dim) {
  int d0_stride = hidden_dim * seq_len;
  int d1_stride = d0_stride / head_num;
  int d2_stride = hidden_dim / head_num;

  int d0_out_stride = d0_stride;
  int d1_out_stride = d2_stride;
  int d2_out_stride = hidden_dim;

  int d0 = blockIdx.x;            // Batch
  int d1 = blockIdx.y / seq_len;  // Head
  int d2 = blockIdx.y % seq_len;  // Sequence Id
  int d3 = threadIdx.x;           // Values

  if (d2 < seq_len) {
    float val_q = Q[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
    float val_k = K[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
    q_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = val_q;
    k_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = val_k;
  }
}

template <typename T>
MultiHeadAttentionLayer<T>::MultiHeadAttentionLayer(
    const Tensors2<T>& in_tensors, Tensor2<T>& out_tensor,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff, int num_attention_heads,
    const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision,
    bool enable_tf32_compute)
    : Layer(gpu_resource),
      use_mixed_precision_(use_mixed_precision),
      enable_tf32_compute_(enable_tf32_compute) {
  try {
    num_ = in_tensors.size();

    // error input checking
    dims_ = in_tensors[0].get_dimensions().size();
    if (dims_ != 4 && dims_ != 3) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "MultiHeadAttentionLayer needs 4D or 3D input tensors, but accept " +
                         std::to_string(dims_) + "D inputs");
    }
    if (num_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "MultiHeadAttentionLayer needs 2 input tensors: query and key");
    }
    if (in_tensors[1].get_dimensions().size() != dims_) {
      HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
    }
    if (dims_ == 4 &&
        in_tensors[1].get_dimensions()[dims_ - 1] != in_tensors[0].get_dimensions()[dims_ - 1]) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "The last two dimension of 4D the input tensors should be m x n, k x n");
    }
    if (dims_ == 3 &&
        in_tensors[0].get_dimensions()[dims_ - 1] != in_tensors[1].get_dimensions()[dims_ - 1]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "3D input tensors must have the same hidden_dim");
    }
    if (dims_ == 3 &&
        in_tensors[0].get_dimensions()[dims_ - 2] != in_tensors[1].get_dimensions()[dims_ - 2]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "3D input tensors must have the same seq_len");
    }
    if (in_tensors[0].get_dimensions()[0] != in_tensors[1].get_dimensions()[0]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "input tensors must have the same batch_size");
    }

    for (size_t i = 0; i < num_; i++) {
      in_tensors_.push_back(in_tensors[i]);
    }
    size_t m = 0, k = 0, h = 0, b = 0, size_per_head = 0;
    if (dims_ == 4) {
      m = in_tensors[0].get_dimensions()[dims_ - 2];
      k = in_tensors[1].get_dimensions()[dims_ - 2];

      h = in_tensors[0].get_dimensions()[1];
      b = in_tensors[0].get_dimensions()[0];
      size_per_head = in_tensors[0].get_dimensions()[dims_ - 1];
    }
    if (dims_ == 3) {
      m = in_tensors[0].get_dimensions()[dims_ - 2];
      k = in_tensors[1].get_dimensions()[dims_ - 2];

      h = num_attention_heads;
      b = in_tensors[0].get_dimensions()[0];
      size_per_head = in_tensors[0].get_dimensions()[dims_ - 1] / h;
    }
    num_head_ = h;
    std::vector<size_t> out_dim = {b, h, m, k};
    blobs_buff->reserve(out_dim, &out_tensor);

    out_tensor_ = out_tensor;

    blobs_buff->reserve(in_tensors[0].get_dimensions(), &fprop_inputA_);
    blobs_buff->reserve({b, h, m, size_per_head}, &query_buf_);
    blobs_buff->reserve({b, h, m, size_per_head}, &key_buf_);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void MultiHeadAttentionLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  T* query = in_tensors_[0].get_ptr();
  T* query_buf = query_buf_.get_ptr();
  T* key = in_tensors_[1].get_ptr();
  T* key_buf = key_buf_.get_ptr();
  T* out = out_tensor_.get_ptr();

  const auto& in_tensor_dim = in_tensors_[0].get_dimensions();
  const auto& out_tensor_dim = out_tensor_.get_dimensions();

  size_t head_num = 0, batch_size = 0, from_seq_len = 0, to_seq_len = 0, size_per_head = 0;
  if (dims_ == 4) {
    head_num = in_tensor_dim[1];
    batch_size = in_tensor_dim[0];
    from_seq_len = in_tensor_dim[dims_ - 2];
    size_per_head = in_tensor_dim[dims_ - 1];
    to_seq_len = out_tensor_dim[dims_ - 1];
  }

  if (dims_ == 3) {
    head_num = num_head_;
    batch_size = in_tensor_dim[0];
    from_seq_len = in_tensor_dim[dims_ - 2];
    to_seq_len = from_seq_len;
    size_per_head = in_tensor_dim[dims_ - 1] / head_num;
    dim3 block_dim(size_per_head, head_num);
    dim3 grid_dim(batch_size, from_seq_len);
    transpose_QK<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
        query_buf, key_buf, query, key, batch_size, from_seq_len, head_num,
        head_num * size_per_head);
    query = query_buf;
    key = key_buf;
  }

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
  T* query_buf = query_buf_.get_ptr();
  T* key = in_tensors_[1].get_ptr();
  T* key_buf = key_buf_.get_ptr();
  T* out = out_tensor_.get_ptr();

  const auto& in_tensor_dim = in_tensors_[0].get_dimensions();
  const auto& out_tensor_dim = out_tensor_.get_dimensions();

  size_t head_num = 0, batch_size = 0, from_seq_len = 0, to_seq_len = 0, size_per_head = 0;
  if (dims_ == 4) {
    head_num = in_tensor_dim[1];
    batch_size = in_tensor_dim[0];
    from_seq_len = in_tensor_dim[dims_ - 2];
    size_per_head = in_tensor_dim[dims_ - 1];
    to_seq_len = out_tensor_dim[dims_ - 1];
  }

  if (dims_ == 3) {
    head_num = num_head_;
    batch_size = in_tensor_dim[0];
    from_seq_len = in_tensor_dim[dims_ - 2];
    size_per_head = in_tensor_dim[dims_ - 1] / head_num;
    /*dim3 block_dim(size_per_head, head_num);
    dim3 grid_dim(batch_size, from_seq_len);
    transpose_QK<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
        query_buf, key_buf, query, key, batch_size, from_seq_len, head_num,
        head_num * size_per_head);*/
    query = query_buf;
    key = key_buf;
  }

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

  if (dims_ == 3) {
    query = in_tensors_[0].get_ptr();
    query_buf = query_buf_.get_ptr();
    key = in_tensors_[1].get_ptr();
    key_buf = key_buf_.get_ptr();
    dim3 block_dim(size_per_head * head_num);
    dim3 grid_dim(batch_size, head_num * from_seq_len);
    transpose_QK_back<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
        query, key, query_buf, key_buf, batch_size, from_seq_len, head_num,
        head_num * size_per_head);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class MultiHeadAttentionLayer<float>;
template class MultiHeadAttentionLayer<__half>;

}  // namespace HugeCTR

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
__global__ void transpose_QKV(T* q_buf, T* k_buf, T* v_buf, const T* Q, const T* K, const T* V,
                              const int batch_size, const int seq_len, const int head_num,
                              const int hidden_dim) {
  int d0_stride = hidden_dim * seq_len;
  int d1_stride = hidden_dim;
  int d2_stride = hidden_dim / head_num;

  int d0_out_stride = d0_stride;
  int d1_out_stride = d2_stride;
  int d2_out_stride = d2_stride * seq_len;

  int d0 = blockIdx.x;   // Batch
  int d1 = blockIdx.y;   // Sequence ID (0-127)
  int d2 = threadIdx.y;  // Head (0-11)
  int d3 = threadIdx.x;  // Values

  float input_Q = Q[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
  q_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = input_Q;

  float input_K = K[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
  k_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = input_K;

  float input_V = V[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
  v_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = input_V;
}
template <typename T>
__global__ void transpose_V(T* v_buf, const T* V, const int batch_size, const int seq_len,
                            const int head_num, const int hidden_dim) {
  int d0_stride = hidden_dim * seq_len;
  int d1_stride = hidden_dim;
  int d2_stride = hidden_dim / head_num;

  int d0_out_stride = d0_stride;
  int d1_out_stride = d2_stride;
  int d2_out_stride = d2_stride * seq_len;

  int d0 = blockIdx.x;   // Batch
  int d1 = blockIdx.y;   // Sequence ID (0-127)
  int d2 = threadIdx.y;  // Head (0-11)
  int d3 = threadIdx.x;  // Values

  float input_V = V[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
  v_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = input_V;
}
// transpose from [batch_size, head_num, seq_len, size_per_head] -> [batch_size, seq_len,
// hidden_num]
template <typename T>
__global__ void transpose_QKV_back(T* q_buf, T* k_buf, T* v_buf, const T* Q, const T* K, const T* V,
                                   const int batch_size, const int seq_len, const int head_num,
                                   const int hidden_dim) {
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
    float val_v = V[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
    q_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = val_q;
    k_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = val_k;
    v_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = val_v;
  }
}

template <typename T>
__global__ void transpose_V_back(T* v_buf, const T* V, const int batch_size, const int seq_len,
                                 const int head_num, const int hidden_dim) {
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
    float val_v = V[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
    v_buf[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = val_v;
  }
}

template <typename T>
MultiHeadAttentionLayer<T>::MultiHeadAttentionLayer(
    const Tensors2<T>& in_tensors, Tensors2<T>& out_tensors,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff, int num_attention_heads,
    bool transpose_b, const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision,
    bool enable_tf32_compute)
    : Layer(gpu_resource),
      use_mixed_precision_(use_mixed_precision),
      enable_tf32_compute_(enable_tf32_compute) {
  try {
    transpose_b_ = transpose_b;
    num_ = in_tensors.size();

    // error input checking
    dims_ = in_tensors[0].get_dimensions().size();
    if (dims_ != 4 && dims_ != 3) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "MultiHeadAttentionLayer needs 4D or 3D input tensors, but accept " +
                         std::to_string(dims_) + "D inputs");
    }
    if (dims_ == 4) {
      if (num_ < 2) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "MultiHeadAttentionLayer needs 2 input tensors: query and key");
      }
      if (in_tensors[1].get_dimensions().size() != dims_) {
        HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
      }
      if (in_tensors[0].get_dimensions()[0] != in_tensors[1].get_dimensions()[0]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "input tensors must have the same batch_size");
      }
      if (in_tensors[1].get_dimensions()[dims_ - 1] != in_tensors[0].get_dimensions()[dims_ - 1] &&
          in_tensors[0].get_dimensions()[dims_ - 1] != in_tensors[1].get_dimensions()[dims_ - 2]) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "The last two dimension of 4D the input tensors should be m x n, k x n or m "
                       "x n, n x k");
      }
    }
    if (dims_ == 3) {
      // query: [batch_size, seq_len, hidden_dim]
      // key: [batch_size, seq_len, hidden_dim]
      // value: [batch_size, seq_len, hidden_dim]
      if (num_ < 3) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "MultiHeadAttentionLayer needs 3 input tensors: query, key and value");
      }
      if (in_tensors[1].get_dimensions().size() != dims_ ||
          in_tensors[2].get_dimensions().size() != dims_) {
        HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
      }
      if (in_tensors[0].get_dimensions()[dims_ - 1] != in_tensors[1].get_dimensions()[dims_ - 1] ||
          in_tensors[0].get_dimensions()[dims_ - 1] != in_tensors[2].get_dimensions()[dims_ - 1]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "3D input tensors must have the same hidden_dim");
      }
      if (in_tensors[0].get_dimensions()[dims_ - 2] != in_tensors[1].get_dimensions()[dims_ - 2] ||
          in_tensors[0].get_dimensions()[dims_ - 2] != in_tensors[2].get_dimensions()[dims_ - 2]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "3D input tensors must have the same seq_len");
      }
    }

    for (size_t i = 0; i < num_; i++) {
      in_tensors_.push_back(in_tensors[i]);
    }
    size_t m = 0, k = 0, h = 0, b = 0, size_per_head = 0;
    if (dims_ == 4 && transpose_b_) {
      b = in_tensors[0].get_dimensions()[0];
      h = in_tensors[0].get_dimensions()[1];
      m = in_tensors[0].get_dimensions()[dims_ - 2];
      k = in_tensors[1].get_dimensions()[dims_ - 2];
      size_per_head = in_tensors[0].get_dimensions()[dims_ - 1];
    }
    if (dims_ == 4 && !transpose_b_) {
      b = in_tensors[0].get_dimensions()[0];
      h = in_tensors[0].get_dimensions()[1];

      m = in_tensors[0].get_dimensions()[dims_ - 2];
      k = in_tensors[0].get_dimensions()[dims_ - 1];
      size_per_head = in_tensors[1].get_dimensions()[dims_ - 1];
    }
    if (dims_ == 3) {
      transpose_b_ = true;
      b = in_tensors[0].get_dimensions()[0];
      h = num_attention_heads;

      m = in_tensors[0].get_dimensions()[dims_ - 2];
      k = in_tensors[1].get_dimensions()[dims_ - 2];
      size_per_head = in_tensors[0].get_dimensions()[dims_ - 1] / h;
    }
    num_head_ = h;
    if (transpose_b_) {
      std::vector<size_t> out_dim = {b, h, m, k};

      Tensor2<T> attention_score_item;
      blobs_buff->reserve(out_dim, &attention_score_item);
      out_tensors.push_back(attention_score_item);
    } else {
      blobs_buff->reserve({b, h, m, size_per_head}, &value_buf_);
      std::vector<size_t> out_dim = {b, m, size_per_head * h};

      Tensor2<T> attention_out_item;
      blobs_buff->reserve(out_dim, &attention_out_item);
      out_tensors.push_back(attention_out_item);
    }

    blobs_buff->reserve(in_tensors[0].get_dimensions(), &fprop_inputA_);
    if (dims_ == 3) {
      blobs_buff->reserve({b, h, m, size_per_head}, &query_buf_);
      blobs_buff->reserve({b, h, m, size_per_head}, &key_buf_);
      Tensor2<T> value_4d_item;
      blobs_buff->reserve({b, h, m, size_per_head}, &value_4d_item);
      out_tensors.push_back(value_4d_item);
    }

    for (auto& out_tensor : out_tensors) {
      out_tensors_.push_back(out_tensor);
    }
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}  // namespace HugeCTR

template <typename T>
void MultiHeadAttentionLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  T* query = in_tensors_[0].get_ptr();
  T* query_buf;
  T* key = in_tensors_[1].get_ptr();
  T* key_buf;
  T* value;
  T* score;
  T* value_4d;
  T* attention_out = out_tensors_[0].get_ptr();

  const auto& in_tensor_dim = in_tensors_[0].get_dimensions();
  const auto& out_tensor_dim = out_tensors_[0].get_dimensions();

  size_t head_num = 0, batch_size = 0, from_seq_len = 0, to_seq_len = 0, size_per_head = 0;
  if (dims_ == 4 && transpose_b_) {
    head_num = in_tensor_dim[1];
    batch_size = in_tensor_dim[0];
    from_seq_len = in_tensor_dim[dims_ - 2];
    size_per_head = in_tensor_dim[dims_ - 1];
    to_seq_len = out_tensor_dim[dims_ - 1];
  }
  if (dims_ == 4 && !transpose_b_) {
    batch_size = in_tensor_dim[0];
    head_num = in_tensor_dim[1];
    from_seq_len = in_tensor_dim[dims_ - 2];
    to_seq_len = in_tensor_dim[dims_ - 1];
    size_per_head = in_tensors_[1].get_dimensions()[dims_ - 1];
  }

  if (dims_ == 3) {
    query_buf = query_buf_.get_ptr();
    key_buf = key_buf_.get_ptr();
    value = in_tensors_[2].get_ptr();
    value_4d = out_tensors_[1].get_ptr();
    head_num = num_head_;
    batch_size = in_tensor_dim[0];
    from_seq_len = in_tensor_dim[dims_ - 2];
    to_seq_len = from_seq_len;
    size_per_head = in_tensor_dim[dims_ - 1] / head_num;
    dim3 block_dim(size_per_head, head_num);
    dim3 grid_dim(batch_size, from_seq_len);
    transpose_QKV<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
        query_buf, key_buf, value_4d, query, key, value, batch_size, from_seq_len, head_num,
        head_num * size_per_head);
    query = query_buf;
    key = key_buf;
  }
  const int batch_count = head_num * batch_size;
  cudaDataType_t a_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;
  cudaDataType_t b_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;
  cudaDataType_t c_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;

  cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  cublasGemmAlgo_t algo =
      use_mixed_precision_ ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;

  if (transpose_b_) {
    const int m = from_seq_len;
    const int n = to_seq_len;
    const int k = size_per_head;
    float alpha = 1.0f / (float)(sqrt(size_per_head)), beta = 0.0f;
    long long int stride_a = static_cast<long long int>(to_seq_len) * size_per_head;
    long long int stride_b = static_cast<long long int>(from_seq_len) * size_per_head;
    long long int stride_c = static_cast<long long int>(from_seq_len) * to_seq_len;

    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, key, a_type, k,
        stride_a, query, b_type, k, stride_b, &beta, attention_out, c_type, n, stride_c,
        batch_count, compute_type, algo));

  } else {
    score = in_tensors_[0].get_ptr();
    value = in_tensors_[1].get_ptr();
    value_4d = value_buf_.get_ptr();
    const int m = from_seq_len;
    const int n = size_per_head;
    const int k = to_seq_len;

    float alpha = 1.0f, beta = 0.0f;
    long long int stride_a = static_cast<long long int>(m) * k;
    long long int stride_b = static_cast<long long int>(k) * n;
    long long int stride_c = static_cast<long long int>(m) * n;

    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, value, b_type, n,
        stride_b, score, a_type, k, stride_a, &beta, value_4d, c_type, n, stride_c, batch_count,
        compute_type, algo));
    dim3 block_dim(size_per_head);
    dim3 grid_dim(batch_size, head_num * from_seq_len);
    transpose_V_back<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
        attention_out, value_4d, batch_size, from_seq_len, head_num, head_num * size_per_head);
  }
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
  T* query_buf;
  T* key = in_tensors_[1].get_ptr();
  T* key_buf;
  T* value;
  T* score;
  T* value_4d;
  T* attention_out = out_tensors_[0].get_ptr();

  const auto& in_tensor_dim = in_tensors_[0].get_dimensions();
  const auto& out_tensor_dim = out_tensors_[0].get_dimensions();

  size_t head_num = 0, batch_size = 0, from_seq_len = 0, to_seq_len = 0, size_per_head = 0;
  if (dims_ == 4 && transpose_b_) {
    query = in_tensors_[0].get_ptr();
    key = in_tensors_[1].get_ptr();
    head_num = in_tensor_dim[1];
    batch_size = in_tensor_dim[0];
    from_seq_len = in_tensor_dim[dims_ - 2];
    size_per_head = in_tensor_dim[dims_ - 1];
    to_seq_len = out_tensor_dim[dims_ - 1];
  }

  if (dims_ == 4 && !transpose_b_) {
    batch_size = in_tensor_dim[0];
    head_num = in_tensor_dim[1];
    from_seq_len = in_tensor_dim[dims_ - 2];
    to_seq_len = in_tensor_dim[dims_ - 1];
    size_per_head = in_tensors_[1].get_dimensions()[dims_ - 1];
  }
  if (dims_ == 3) {
    head_num = num_head_;
    batch_size = in_tensor_dim[0];
    from_seq_len = in_tensor_dim[dims_ - 2];
    to_seq_len = in_tensor_dim[dims_ - 2];
    size_per_head = in_tensor_dim[dims_ - 1] / head_num;
    query = query_buf_.get_ptr();
    key = key_buf_.get_ptr();
  }

  const int batch_count = head_num * batch_size;
  cudaDataType_t a_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;
  cudaDataType_t b_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;
  cudaDataType_t c_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;

  cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  cublasGemmAlgo_t algo =
      use_mixed_precision_ ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
  if (transpose_b_) {
    const int m = from_seq_len;
    const int n = size_per_head;
    const int k = to_seq_len;

    float alpha = 1.0f / (float)(sqrt(size_per_head)), beta = 0.0f;

    long long int stride_a = static_cast<long long int>(n) * k;
    long long int stride_b = static_cast<long long int>(k) * m;
    long long int stride_c = static_cast<long long int>(n) * m;

    // gradient respect to input A   matmul(C,B)
    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, key, a_type, n,
        stride_a, attention_out, b_type, k, stride_b, &beta, query, c_type, n, stride_c,
        batch_count, compute_type, algo));

    T* cur_Q = fprop_inputA_.get_ptr();

    // gradient respect to input B  matmul(C^T,A)
    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, cur_Q, a_type, n,
        stride_c, attention_out, b_type, k, stride_b, &beta, key, c_type, n, stride_a, batch_count,
        compute_type, algo));

    if (dims_ == 3) {
      query = in_tensors_[0].get_ptr();
      query_buf = query_buf_.get_ptr();
      key = in_tensors_[1].get_ptr();
      key_buf = key_buf_.get_ptr();
      value = in_tensors_[2].get_ptr();
      value_4d = out_tensors_[1].get_ptr();
      dim3 block_dim(size_per_head);
      dim3 grid_dim(batch_size, head_num * from_seq_len);
      transpose_QKV_back<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
          query, key, value, query_buf, key_buf, value_4d, batch_size, from_seq_len, head_num,
          head_num * size_per_head);
    }
  } else {
    score = in_tensors_[0].get_ptr();
    value = in_tensors_[1].get_ptr();
    value_4d = value_buf_.get_ptr();
    dim3 block_dim(size_per_head, head_num);
    dim3 grid_dim(batch_size, from_seq_len);
    transpose_V<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
        value_4d, attention_out, batch_size, from_seq_len, head_num, head_num * size_per_head);
    const int m = from_seq_len;
    const int n = to_seq_len;
    const int k = size_per_head;

    float alpha = 1.0f, beta = 0.0f;

    long long int stride_a = static_cast<long long int>(n) * k;
    long long int stride_b = static_cast<long long int>(k) * m;
    long long int stride_c = static_cast<long long int>(n) * m;

    // gradient respect to input A   matmul(C,B^T)
    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, value, b_type, k,
        stride_a, value_4d, b_type, k, stride_b, &beta, score, c_type, n, stride_c, batch_count,
        compute_type, algo));
    T* cur_Q = fprop_inputA_.get_ptr();

    // gradient respect to input B     matmul(A^T, C)
    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, k, n, m, &alpha, value_4d, a_type,
        k, stride_b, cur_Q, b_type, n, stride_c, &beta, value, c_type, k, stride_a, batch_count,
        compute_type, algo));
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class MultiHeadAttentionLayer<float>;
template class MultiHeadAttentionLayer<__half>;

}  // namespace HugeCTR

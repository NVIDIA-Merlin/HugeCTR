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

#include <algorithm>
#include <core23/tensor_operations.hpp>
#include <functional>
#include <layers/multi_head_attention_layer.hpp>
#include <memory>
#include <network_buffer_channels.hpp>
#include <utils.cuh>
#include <utils.hpp>
namespace HugeCTR {

// Q ==> seq_out
// K,V ==> seq_in
// transpose
// from [batch_size, seq_len, hidden_num] => [batch_size, seq_len, head_num, size_per_head] =>
// to [batch_size, head_num, seq_len, size_per_head]
template <typename T>
__global__ void transpose_0213(T* output, const T* input, const int batch_size, const int seq_len,
                               const int head_num, const int size_per_head) {
  const int hidden_dim = size_per_head * head_num;
  int d0_stride = hidden_dim * seq_len;
  int d1_stride = hidden_dim;
  int d2_stride = size_per_head;

  int d0_out_stride = d0_stride;
  int d1_out_stride = d2_stride;
  int d2_out_stride = d2_stride * seq_len;
  int flatten_len = batch_size * seq_len * head_num * size_per_head;
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  for (int in_idx = gtid; in_idx < flatten_len; in_idx += grid_stride) {
    int d0 = in_idx / d0_stride;                // Batch
    int d1 = (in_idx % d0_stride) / d1_stride;  // Sequence ID (0-127)
    int d2 = (in_idx % d1_stride) / d2_stride;  // Head (0-11)
    int d3 = in_idx % d2_stride;                // Values
    int out_idx = d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3;
    output[out_idx] = input[in_idx];
  }
}
// 2 arrays
template <typename T>
__global__ void transpose_0213(T* k_buf, T* v_buf, const T* K, const T* V, const int batch_size,
                               const int seq_len, const int head_num, const int size_per_head) {
  const int hidden_dim = size_per_head * head_num;
  int d0_stride = hidden_dim * seq_len;
  int d1_stride = hidden_dim;
  int d2_stride = size_per_head;

  int d0_out_stride = d0_stride;
  int d1_out_stride = d2_stride;
  int d2_out_stride = d2_stride * seq_len;
  int flatten_len = batch_size * seq_len * head_num * size_per_head;
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  for (int in_idx = gtid; in_idx < flatten_len; in_idx += grid_stride) {
    int d0 = in_idx / d0_stride;                // Batch
    int d1 = (in_idx % d0_stride) / d1_stride;  // Sequence ID (0-127)
    int d2 = (in_idx % d1_stride) / d2_stride;  // Head (0-11)
    int d3 = in_idx % d2_stride;                // Values
    int out_idx = d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3;
    v_buf[out_idx] = V[in_idx];
    k_buf[out_idx] = K[in_idx];
  }
}
// 3 arrays
template <typename T>
__global__ void transpose_0213(T* q_buf, T* k_buf, T* v_buf, const T* Q, const T* K, const T* V,
                               const int batch_size, const int seq_len, const int head_num,
                               const int size_per_head) {
  const int hidden_dim = size_per_head * head_num;
  int d0_stride = hidden_dim * seq_len;
  int d1_stride = hidden_dim;
  int d2_stride = size_per_head;

  int d0_out_stride = d0_stride;
  int d1_out_stride = d2_stride;
  int d2_out_stride = d2_stride * seq_len;
  int flatten_len = batch_size * seq_len * head_num * size_per_head;
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  for (int in_idx = gtid; in_idx < flatten_len; in_idx += grid_stride) {
    int d0 = in_idx / d0_stride;                // Batch
    int d1 = (in_idx % d0_stride) / d1_stride;  // Sequence ID (0-127)
    int d2 = (in_idx % d1_stride) / d2_stride;  // Head (0-11)
    int d3 = in_idx % d2_stride;                // Values
    int out_idx = d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3;
    v_buf[out_idx] = V[in_idx];
    k_buf[out_idx] = K[in_idx];
    q_buf[out_idx] = Q[in_idx];
  }
}

// transpose from [batch_size, head_num, seq_len, size_per_head] -> [batch_size, seq_len,
// hidden_num]
// TODO remove it if perf is not good
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
// TODO remove it if perf is not good
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
// input is q, k, v, mask
template <typename T>
MultiHeadAttentionLayer<T>::MultiHeadAttentionLayer(
    const std::vector<core23::Tensor>& input_tensors, std::vector<core23::Tensor>& output_tensors,
    int num_attention_heads, bool transpose_b, const std::shared_ptr<GPUResource>& gpu_resource,
    bool use_mixed_precision, bool enable_tf32_compute)
    : Layer(input_tensors, {}, gpu_resource),
      use_mixed_precision_(use_mixed_precision),
      enable_tf32_compute_(enable_tf32_compute),
      num_(input_tensors_.size()),
      dims_(input_tensors_[0].dims()) {
  try {
    // k always is the gemm K
    int64_t m = 0, k = 0, h = 0, b = 0, size_per_head = 0;
    for (auto i{0ul}; i < input_tensors.size(); i++) {
      if (i == 3) {
        HCTR_CHECK_HINT(input_tensors[i].dims() == 4, "mask should be 4D tensor");
      } else {
        HCTR_CHECK_HINT(input_tensors[i].dims() == 3, "Query, Key, Value should be 3D tensor");
      }
    }

    // [batchsize, seq_from, hidden]
    auto q_tensor = input_tensors_[0];
    // [batchsize, seq_to, hidden]
    auto k_tensor = input_tensors_[1];
    // [batchsize, seq_to, hidden]
    auto v_tensor = input_tensors_[2];
    auto q_shape = q_tensor.shape();
    auto k_shape = k_tensor.shape();
    auto v_shape = v_tensor.shape();

    HCTR_CHECK_HINT(q_shape[0] == k_shape[0] && k_shape[0] == v_shape[0],
                    "The first dim of Query, Key, Value should be batchsize ");

    HCTR_CHECK_HINT(q_shape[2] == k_shape[2] && k_shape[2] == v_shape[2],
                    " Query, Key, Value should have the same hidden dimension");
    HCTR_CHECK_HINT(k_shape[1] == v_shape[1], "Key, Value should have the same shape");

    transpose_b_ = true;
    b = q_shape[0];
    h = num_attention_heads;
    // m is seq_from
    m = q_shape[1];
    // k is seq_to
    k = k_shape[1];
    size_per_head = q_shape[2] / h;
    num_head_ = h;
    core23::BufferParams buf_p{.channel = GetBlobsBufferChannel()};

    auto common_tensor_params = input_tensors_[0]
                                    .my_params()
                                    .data_type(core23::ToScalarType<T>::value)
                                    .buffer_params(buf_p);
    // this is the bgemm results
    // m is seq_from, k is seq_to
    core23::Shape score_shape = {b, h, m, k};
    core23::Shape from_shape = {b, h, m, size_per_head};
    core23::Shape to_shape = {b, h, k, size_per_head};
    attention_score_4d_ = core23::Tensor(common_tensor_params.shape(score_shape));
    attention_softmax_4d_ = core23::Tensor(common_tensor_params.shape(score_shape));
    attention_out_4d_ = core23::Tensor(common_tensor_params.shape(from_shape));
    output_tensors.emplace_back(common_tensor_params.shape({b, m, size_per_head * h}));

    fprop_query_tensor_ = core23::Tensor(common_tensor_params);
    fprop_softmax_tensor_ = core23::Tensor(common_tensor_params.shape(score_shape));

    query_buf_tensor_ = core23::Tensor(common_tensor_params.shape(from_shape));
    key_buf_tensor_ = core23::Tensor(common_tensor_params.shape(to_shape));
    attention_value_4d_ = core23::Tensor(common_tensor_params.shape(to_shape));

    output_tensors_ = output_tensors;
    // with masked
    if (input_tensors.size() == 4) {
      std::vector<core23::Tensor> bottoms{attention_score_4d_, input_tensors[3]};
      masked_softmax_layer_ = std::make_unique<MaskedSoftmaxLayer<T>>(
          bottoms, attention_softmax_4d_, 1.0f, gpu_resource);
      masked_softmax_layer_->initialize();
    } else {
      softmax_layer_ = std::make_unique<SoftmaxLayer<T>>(attention_score_4d_, attention_softmax_4d_,
                                                         gpu_resource);
      softmax_layer_->initialize();
    }
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void MultiHeadAttentionLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  T* query = input_tensors_[0].data<T>();
  T* key = input_tensors_[1].data<T>();
  T* value = input_tensors_[2].data<T>();

  T* query_buf = query_buf_tensor_.data<T>();
  T* key_buf = key_buf_tensor_.data<T>();

  T* score = attention_score_4d_.data<T>();
  T* value_4d = attention_value_4d_.data<T>();
  // attention_out = transpose(attention_out_tmp)
  T* attention_out = output_tensors_[0].data<T>();
  T* attention_out_tmp = attention_out_4d_.data<T>();

  const auto& in_tensor_shape = input_tensors_[0].shape();
  const auto& out_tensor_shape = output_tensors_[0].shape();

  size_t head_num = 0, batch_size = 0, from_seq_len = 0, to_seq_len = 0, size_per_head = 0;
  /* 1. transpose
      input : q,k,v
      output : query_buf, key_buf, attention_value_4d_(value)
  */
  {
    head_num = num_head_;
    batch_size = in_tensor_shape[0];
    from_seq_len = in_tensor_shape[dims_ - 2];
    to_seq_len = input_tensors_[1].size(dims_ - 2);
    size_per_head = in_tensor_shape[dims_ - 1] / head_num;

    if (from_seq_len == to_seq_len) {
      int flatten_len = batch_size * head_num * size_per_head * from_seq_len;
      dim3 block_dim(1024);
      dim3 grid_dim((flatten_len - 1) / block_dim.x + 1);
      transpose_0213<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
          query_buf, key_buf, value_4d, query, key, value, batch_size, from_seq_len, head_num,
          size_per_head);
    } else {
      dim3 block_dim(1024);
      int len = batch_size * head_num * size_per_head * from_seq_len;
      dim3 grid_dim((len - 1) / block_dim.x + 1);
      // transpose Q
      transpose_0213<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
          query_buf, query, batch_size, from_seq_len, head_num, size_per_head);
      len = batch_size * head_num * size_per_head * to_seq_len;
      grid_dim.x = (len - 1) / block_dim.x + 1;
      // transpose KV
      transpose_0213<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
          key_buf, value_4d, key, value, batch_size, to_seq_len, head_num, size_per_head);
    }
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
  /* 2. bgemm
     input : query_buf, key_buf
     output: attention_score_4d_(score)
  */

  {
    const int m = from_seq_len;
    const int n = to_seq_len;
    const int k = size_per_head;
    float alpha = 1.0f / (float)(sqrt(size_per_head)), beta = 0.0f;
    long long int stride_a = static_cast<long long int>(to_seq_len) * size_per_head;
    long long int stride_b = static_cast<long long int>(from_seq_len) * size_per_head;
    long long int stride_c = static_cast<long long int>(from_seq_len) * to_seq_len;

    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, key, a_type, k,
        stride_a, query, b_type, k, stride_b, &beta, score, c_type, n, stride_c, batch_count,
        compute_type, algo));
  }

  /* 3. softmax
     input : attention_score_4d_
     output: attention_softmax_4d_
  */
  if (masked_softmax_layer_) {
    masked_softmax_layer_->fprop(true);
  } else {
    softmax_layer_->fprop(true);
  }
  score = attention_softmax_4d_.data<T>();
  /* 4. bgemm
     input : attention_softmax_4d_(score), attention_value_4d_
     output: attention_out_tmp -> attention_out
  */
  {
    const int m = from_seq_len;
    const int n = size_per_head;
    const int k = to_seq_len;

    float alpha = 1.0f, beta = 0.0f;
    long long int stride_a = static_cast<long long int>(m) * k;
    long long int stride_b = static_cast<long long int>(k) * n;
    long long int stride_c = static_cast<long long int>(m) * n;

    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, value_4d, b_type,
        n, stride_b, score, a_type, k, stride_a, &beta, attention_out_tmp, c_type, n, stride_c,
        batch_count, compute_type, algo));
    {
      int flatten_len = batch_size * head_num * size_per_head * from_seq_len;
      dim3 block_dim(1024);
      dim3 grid_dim((flatten_len - 1) / block_dim.x + 1);
      transpose_0213<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
          attention_out, attention_out_tmp, batch_size, head_num, from_seq_len, size_per_head);
    }
  }
  HCTR_LIB_THROW(cudaMemcpyAsync(fprop_query_tensor_.data(), (void*)query_buf,
                                 input_tensors_[0].num_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
  HCTR_LIB_THROW(cudaMemcpyAsync(fprop_softmax_tensor_.data(), attention_softmax_4d_.data(),
                                 attention_softmax_4d_.num_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
}

template <typename T>
void MultiHeadAttentionLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  T* query = input_tensors_[0].data<T>();
  T* key = input_tensors_[1].data<T>();
  T* value = input_tensors_[2].data<T>();

  T* query_buf = query_buf_tensor_.data<T>();
  T* key_buf = key_buf_tensor_.data<T>();

  T* score = attention_softmax_4d_.data<T>();
  T* value_4d = attention_value_4d_.data<T>();
  T* attention_out = output_tensors_[0].data<T>();
  T* attention_out_tmp = attention_out_4d_.data<T>();

  const auto& in_tensor_shape = input_tensors_[0].shape();
  const auto& out_tensor_shape = output_tensors_[0].shape();

  size_t head_num = 0, batch_size = 0, from_seq_len = 0, to_seq_len = 0, size_per_head = 0;
  {
    head_num = num_head_;
    batch_size = in_tensor_shape[0];
    from_seq_len = in_tensor_shape[dims_ - 2];
    to_seq_len = input_tensors_[1].size(dims_ - 2);
    size_per_head = in_tensor_shape[dims_ - 1] / head_num;
  }

  const int batch_count = head_num * batch_size;
  cudaDataType_t a_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;
  cudaDataType_t b_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;
  cudaDataType_t c_type = use_mixed_precision_ ? CUDA_R_16F : CUDA_R_32F;

  cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  cublasGemmAlgo_t algo =
      use_mixed_precision_ ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
  /* 1. bgemm
     input : attention_out -> attention_out_tmp
     output: attention_softmax_4d_(score), attention_value_4d_
  */
  {
    int flatten_len = batch_size * head_num * size_per_head * from_seq_len;
    dim3 block_dim(1024);
    dim3 grid_dim((flatten_len - 1) / block_dim.x + 1);

    transpose_0213<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
        attention_out_tmp, attention_out, batch_size, from_seq_len, head_num, size_per_head);
    const int m = from_seq_len;
    const int n = to_seq_len;
    const int k = size_per_head;

    float alpha = 1.0f, beta = 0.0f;

    long long int stride_a = static_cast<long long int>(n) * k;
    long long int stride_b = static_cast<long long int>(k) * m;
    long long int stride_c = static_cast<long long int>(n) * m;

    // gradient respect to input A   matmul(C,B^T)
    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, value_4d, b_type,
        k, stride_a, attention_out_tmp, b_type, k, stride_b, &beta, score, c_type, n, stride_c,
        batch_count, compute_type, algo));
    T* cur_Q = fprop_softmax_tensor_.data<T>();

    // gradient respect to input B     matmul(A^T, C)
    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, k, n, m, &alpha, attention_out_tmp,
        a_type, k, stride_b, cur_Q, b_type, n, stride_c, &beta, value_4d, c_type, k, stride_a,
        batch_count, compute_type, algo));
  }
  /* 2. softmax
     input:  attention_softmax_4d_
     output: attention_score_4d_
  */

  if (masked_softmax_layer_) {
    masked_softmax_layer_->bprop();
  } else {
    softmax_layer_->bprop();
  }
  score = attention_score_4d_.data<T>();

  /* 3. bgemm
     input: attention_score_4d_
     output : query_buf, key_buf
  */
  {
    const int m = from_seq_len;
    const int n = size_per_head;
    const int k = to_seq_len;

    float alpha = 1.0f / (float)(sqrt(size_per_head)), beta = 0.0f;

    long long int stride_a = static_cast<long long int>(n) * k;
    long long int stride_b = static_cast<long long int>(k) * m;
    long long int stride_c = static_cast<long long int>(n) * m;

    // gradient respect to input A   matmul(C,B)
    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, key_buf, a_type,
        n, stride_a, score, b_type, k, stride_b, &beta, query_buf, c_type, n, stride_c, batch_count,
        compute_type, algo));
    // a copy of query_buf
    T* cur_Q = fprop_query_tensor_.data<T>();

    // gradient respect to input B  matmul(C^T,A)
    HCTR_LIB_THROW(cublasGemmStridedBatchedEx(
        get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, cur_Q, a_type, n,
        stride_c, score, b_type, k, stride_b, &beta, key_buf, c_type, n, stride_a, batch_count,
        compute_type, algo));
  }

  /* 4. transpose
    input : query_buf, key_buf, attention_value_4d_(value)
    output: q,k,v
  */
  {
    if (from_seq_len == to_seq_len) {
      int flatten_len = batch_size * head_num * size_per_head * from_seq_len;
      dim3 block_dim(1024);
      dim3 grid_dim((flatten_len - 1) / block_dim.x + 1);
      transpose_0213<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
          query, key, value, query_buf, key_buf, value_4d, batch_size, head_num, from_seq_len,
          size_per_head);
    } else {
      int flatten_len = batch_size * head_num * size_per_head * from_seq_len;
      dim3 block_dim(1024);
      dim3 grid_dim((flatten_len - 1) / block_dim.x + 1);

      // transpose Q
      transpose_0213<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
          query, query_buf, batch_size, head_num, from_seq_len, size_per_head);
      flatten_len = batch_size * head_num * size_per_head * to_seq_len;
      grid_dim.x = (flatten_len - 1) / block_dim.x + 1;
      // transpose KV
      transpose_0213<<<grid_dim, block_dim, 0, get_gpu().get_stream()>>>(
          key, value, key_buf, value_4d, batch_size, head_num, to_seq_len, size_per_head);
    }
  }
}

template class MultiHeadAttentionLayer<float>;
template class MultiHeadAttentionLayer<__half>;

}  // namespace HugeCTR

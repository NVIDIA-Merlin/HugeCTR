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

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <layers/multi_head_attention_layer.hpp>
#include <memory>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {
template <typename T>
void sum_ex_cpu(T *top, int embedding_vector_size, int dim0, T *workspace) {
  // sum(e^xi) i = [0, embedding_vector_size -1];
  for (int i = 0; i < dim0; i++) {
    workspace[i] = (T)0.f;
    int offset = i * embedding_vector_size;
    for (int j = 0; j < embedding_vector_size; j++) {
      workspace[i] = workspace[i] + top[offset + j];
    }
  }
}

template <typename T>
void ex_cpu(T *top, const T *bottom, int len) {
  // e^xi
  for (int i = 0; i < len; i++) {
    top[i] = expf(bottom[i]);
  }
}

template <typename T>
void sum_grad_softmax(const T *d_top, const T *softmax_out, int embedding_vector_size, int dim0,
                      T *workspace) {
  for (int i = 0; i < dim0; i++) {
    float grad_sum = 0.0;
    int offset = i * embedding_vector_size;
    for (int j = 0; j < embedding_vector_size; j++) {
      grad_sum += (float)(d_top[offset + j] * softmax_out[offset + j]);
    }
    workspace[i] = static_cast<T>(grad_sum);
    // printf("CPU grad_sum %d: %f\n", i, workspace[i]);
  }
}

template <typename T>
void softmax_fprop_cpu(T *top, const T *bottom, int len, int embedding_vector_size) {
  int dim0 = len / embedding_vector_size;
  T *workspace = new T[dim0];
  // e^xi
  ex_cpu(top, bottom, len);
  // sum(e^xi) i = [0, embedding_vector_size -1];
  sum_ex_cpu(top, embedding_vector_size, dim0, workspace);
  // softmax : e^xi / sum(e^xi); i = [0, len - 1];
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < embedding_vector_size; j++) {
      int index = i * embedding_vector_size + j;
      top[index] = top[index] / workspace[i];
    }
  }
  delete[] workspace;
}

template <typename T>
void softmax_bprop_cpu(T *d_bottom, const T *d_top, const T *softmax_out, int len,
                       int embedding_vector_size) {
  int dim0 = len / embedding_vector_size;
  T *workspace = new T[dim0];

  sum_grad_softmax(d_top, softmax_out, embedding_vector_size, dim0, workspace);
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < embedding_vector_size; j++) {
      int index = i * embedding_vector_size + j;
      d_bottom[index] = softmax_out[index] * (d_top[index] - workspace[i]);
      // d_bottom[index] = workspace[i];
    }
  }
  delete[] workspace;
}

template <typename T>
T get_eps(bool use_tf32 = false);

template <>
float get_eps(bool use_tf32) {
  return (use_tf32 ? 5e-1 : 1e-3);
}

template <>
__half get_eps(bool use_tf32) {
  return __float2half(1);
}

template <typename T>
void matmul_cpu(T *in1, T *in2, T *output, int64_t B, int64_t H, int64_t M, int64_t N, int64_t K) {
  //(m,n)x(n,k)
  int64_t i, j, k, z, y;
  for (y = 0; y < B; y++) {
    for (z = 0; z < H; z++) {
      for (i = 0; i < M; i++) {
        for (j = 0; j < K; j++) {
          output[y * H * M * K + z * M * K + i * K + j] = TypeConvert<T, float>::convert(0.0f);
          for (k = 0; k < N; k++) {
            output[y * H * M * K + z * M * K + i * K + j] =
                output[y * H * M * K + z * M * K + i * K + j] +
                in1[y * H * M * N + z * M * N + i * N + k] *
                    in2[y * H * N * K + z * N * K + k * K + j];
          }
        }
      }
    }
  }
}

//[b, h, m, n] -> [b, h, n, m]
template <typename T>
static void transpose(T *a, int64_t b, int64_t h, int64_t m, int64_t n) {
  std::unique_ptr<T[]> tmp(new T[m * n]);
  for (int64_t y = 0; y < b; y++) {
    for (int64_t z = 0; z < h; z++) {
      T *cur_a = a + z * m * n + y * h * m * n;
      for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) tmp[j * m + i] = cur_a[i * n + j];
      for (int64_t i = 0; i < m * n; ++i) cur_a[i] = tmp[i];
    }
  }
}

// transpose from [batch_size, seq_len, hidden_num] -> [batch_size, head_num, seq_len,
// size_per_head]
template <typename T>
void transpose_v_cpu(T *value, int64_t batch_size, int64_t seq_len, int64_t head_num,
                     int64_t hidden_dim) {
  auto size = seq_len * hidden_dim;
  auto size_per_head = hidden_dim / head_num;
  std::unique_ptr<T[]> tmp_v(new T[size]);
  for (int64_t b = 0; b < batch_size; b++) {
    T *cur_v = value + b * seq_len * hidden_dim;
    for (int64_t s = 0; s < seq_len; s++) {
      for (int64_t d = 0; d < hidden_dim; d++) {
        tmp_v[(d / size_per_head) * seq_len * size_per_head + s * size_per_head +
              d % size_per_head] = cur_v[s * hidden_dim + d];
      }
    }
    for (int64_t i = 0; i < size; ++i) {
      cur_v[i] = tmp_v[i];
    }
  }
}

template <typename T>
void transpose_qkv_cpu(T *query, T *key, T *value, T *value_4d, int64_t batch_size,
                       int64_t seq_from, int64_t seq_to, int64_t head_num, int64_t hidden_dim) {
  auto qsize = seq_from * hidden_dim;
  auto ksize = seq_to * hidden_dim;
  auto size_per_head = hidden_dim / head_num;
  std::unique_ptr<T[]> tmp_q(new T[qsize]);
  std::unique_ptr<T[]> tmp_k(new T[ksize]);
  for (int64_t b = 0; b < batch_size; b++) {
    T *cur_q = query + b * seq_from * hidden_dim;
    T *cur_k = key + b * seq_to * hidden_dim;
    T *cur_v = value + b * seq_to * hidden_dim;
    T *cur_v_4d = value_4d + b * seq_to * hidden_dim;
    // transpose Q
    for (int64_t s = 0; s < seq_to; s++) {
      for (int64_t d = 0; d < hidden_dim; d++) {
        tmp_k[(d / size_per_head) * seq_to * size_per_head + s * size_per_head +
              d % size_per_head] = cur_k[s * hidden_dim + d];
        cur_v_4d[(d / size_per_head) * seq_to * size_per_head + s * size_per_head +
                 d % size_per_head] = cur_v[s * hidden_dim + d];
      }
    }
    // transpose KV
    for (int64_t s = 0; s < seq_from; s++) {
      for (int64_t d = 0; d < hidden_dim; d++) {
        tmp_q[(d / size_per_head) * seq_from * size_per_head + s * size_per_head +
              d % size_per_head] = cur_q[s * hidden_dim + d];
      }
    }
    for (int64_t i = 0; i < qsize; ++i) {
      cur_q[i] = tmp_q[i];
    }
    for (int64_t i = 0; i < ksize; ++i) {
      cur_k[i] = tmp_k[i];
    }
  }
}

// transpose from [batch_size, head_num, seq_len, size_per_head] -> [batch_size, seq_len,
// hidden_num]
template <typename T>
void transpose_qkv_back_cpu(T *query, T *key, T *value, T *value_4d, int64_t batch_size,
                            int64_t seq_from, int64_t seq_to, int64_t head_num,
                            int64_t hidden_dim) {
  auto qsize = seq_from * hidden_dim;
  auto ksize = seq_to * hidden_dim;
  auto size_per_head = hidden_dim / head_num;
  std::unique_ptr<T[]> tmp_q(new T[qsize]);
  std::unique_ptr<T[]> tmp_k(new T[ksize]);
  for (int64_t b = 0; b < batch_size; b++) {
    T *cur_q = query + b * seq_from * hidden_dim;
    T *cur_k = key + b * seq_to * hidden_dim;
    T *cur_v = value + b * seq_to * hidden_dim;
    T *cur_v_4d = value_4d + b * seq_to * hidden_dim;
    for (int64_t h = 0; h < head_num; h++) {
      // transpose Q
      for (int64_t s = 0; s < seq_from; s++) {
        for (int64_t d = 0; d < size_per_head; d++) {
          tmp_q[s * hidden_dim + h * size_per_head + d] =
              cur_q[h * seq_from * size_per_head + s * size_per_head + d];
        }
      }
      // transpose KV
      for (int64_t s = 0; s < seq_to; s++) {
        for (int64_t d = 0; d < size_per_head; d++) {
          tmp_k[s * hidden_dim + h * size_per_head + d] =
              cur_k[h * seq_to * size_per_head + s * size_per_head + d];
          cur_v[s * hidden_dim + h * size_per_head + d] =
              cur_v_4d[h * seq_to * size_per_head + s * size_per_head + d];
        }
      }
    }
    for (int64_t i = 0; i < qsize; ++i) {
      cur_q[i] = tmp_q[i];
    }
    for (int64_t i = 0; i < ksize; ++i) {
      cur_k[i] = tmp_k[i];
    }
  }
}

// transpose from [batch_size, head_num, seq_len, size_per_head] -> [batch_size, seq_len,
// hidden_num]
template <typename T>
void transpose_v_back_cpu(T *value, int64_t batch_size, int64_t seq_len, int64_t head_num,
                          int64_t hidden_dim) {
  auto size = seq_len * hidden_dim;
  auto size_per_head = hidden_dim / head_num;
  std::unique_ptr<T[]> tmp_v(new T[size]);
  for (int64_t b = 0; b < batch_size; b++) {
    T *cur_v = value + b * seq_len * hidden_dim;
    for (int64_t h = 0; h < head_num; h++) {
      for (int64_t s = 0; s < seq_len; s++) {
        for (int64_t d = 0; d < size_per_head; d++) {
          tmp_v[s * hidden_dim + h * size_per_head + d] =
              cur_v[h * seq_len * size_per_head + s * size_per_head + d];
        }
      }
    }
    for (int64_t i = 0; i < size; ++i) {
      cur_v[i] = tmp_v[i];
    }
  }
}

template <typename T>
void multi_head_attention_cpu(T *in1, T *in2, T *output, int64_t b, int64_t h, int64_t m, int64_t n,
                              int64_t k) {
  transpose(in2, b, h, n, k);
  matmul_cpu(in1, in2, output, b, h, m, k, n);
  // Just to revert in2 back
  transpose(in2, b, h, k, n);
  for (int64_t i = 0; i < b * h * m * n; i++) {
    output[i] = output[i] / ((float)sqrt(k));
  }
}

template <>
void multi_head_attention_cpu(__half *in1, __half *in2, __half *output, int64_t b, int64_t h,
                              int64_t m, int64_t n, int64_t k) {
  transpose(in2, b, h, n, k);
  matmul_cpu(in1, in2, output, b, h, m, k, n);
  // Just to revert in2 back
  transpose(in2, b, h, k, n);
  for (int64_t i = 0; i < b * h * m * n; i++) {
    output[i] = __half2float(output[i]) / ((float)sqrt(k));
  }
}

template <typename T>
void multi_head_attention_cpu_fused(T *q_4d, T *k_4d, T *v_4d, T *output, T *tmp_softmax, int64_t b,
                                    int64_t h, int64_t m, int64_t n, int64_t k) {
  std::unique_ptr<T[]> tmp_score(new T[b * h * m * n]);
  transpose(k_4d, b, h, n, k);
  // q * k
  matmul_cpu(q_4d, k_4d, tmp_score.get(), b, h, m, k, n);
  // Just to revert k_4d back
  transpose(k_4d, b, h, k, n);
  for (int64_t i = 0; i < b * h * m * n; i++) {
    tmp_score[i] = (float)tmp_score[i] / sqrt((float)k);
  }
  // masked softmax
  softmax_fprop_cpu<T>(tmp_softmax, tmp_score.get(), b * h * m * n, n);

  matmul_cpu(tmp_softmax, v_4d, output, b, h, m, n, k);
  transpose_v_back_cpu(output, b, m, h, h * k);
}

template <typename T>
void multi_head_attention_3d_cpu_fused(T *in1, T *in2, T *in3, T *output, T *tmp_softmax,
                                       T *value_out, int64_t batch_size, int64_t seq_from,
                                       int64_t seq_to, int64_t hidden_dim, int64_t head_num) {
  transpose_qkv_cpu(in1, in2, in3, value_out, batch_size, seq_from, seq_to, head_num, hidden_dim);

  multi_head_attention_cpu_fused(in1, in2, value_out, output, tmp_softmax, batch_size, head_num,
                                 seq_from, seq_to, hidden_dim / head_num);
}

template <typename T>
void multi_head_attention_dgrad_cpu(T *out, T **h_ins, T **h_b_ins, int64_t b, int64_t h, int64_t m,
                                    int64_t n, int64_t k) {
  // transpose(h_ins[1], h, b, n, k);
  // transpose(h_ins[0], h, b, m, n);
  // out [b,h,m,n]
  // in1 [b,h,m,k]
  // in2 [b,h,n,k]
  matmul_cpu(out, h_ins[1], h_b_ins[0], b, h, m, n, k);
  transpose(out, b, h, m, n);
  matmul_cpu(out, h_ins[0], h_b_ins[1], b, h, n, m, k);
  // Just revert out back
  transpose(out, b, h, m, n);
  for (int64_t i = 0; i < b * h * m * k; i++) {
    h_b_ins[0][i] = h_b_ins[0][i] / ((float)sqrt(k));
  }
  for (int64_t i = 0; i < b * h * n * k; i++) {
    h_b_ins[1][i] = h_b_ins[1][i] / ((float)sqrt(k));
  }
}

template <>
void multi_head_attention_dgrad_cpu(__half *out, __half **h_ins, __half **h_b_ins, int64_t b,
                                    int64_t h, int64_t m, int64_t n, int64_t k) {
  // transpose(h_ins[1], h, b, n, k);
  // transpose(h_ins[0], h, b, m, n);
  // out [b,h,m,n]
  // in1 [b,h,m,k]
  // in2 [b,h,n,k]
  matmul_cpu(out, h_ins[1], h_b_ins[0], b, h, m, n, k);
  transpose(out, b, h, m, n);
  matmul_cpu(out, h_ins[0], h_b_ins[1], b, h, n, m, k);
  // Just revert out back
  transpose(out, b, h, m, n);
  for (int64_t i = 0; i < b * h * m * k; i++) {
    h_b_ins[0][i] = __half2float(h_b_ins[0][i]) / ((float)sqrt(k));
  }
  for (int64_t i = 0; i < b * h * n * k; i++) {
    h_b_ins[1][i] = __half2float(h_b_ins[1][i]) / ((float)sqrt(k));
  }
}

template <typename T>
void multi_head_attention_dgrad_cpu_fused(T *out, T **h_ins, T **h_b_ins, int64_t b, int64_t h,
                                          int64_t m, int64_t n, int64_t k) {
  // transpose(h_ins[1], h, b, n, k);
  // transpose(h_ins[0], h, b, m, n);
  // out [b,h,m,n]
  // in1 [b,h,m,k]
  // in2 [b,h,n,k]
  matmul_cpu(out, h_ins[1], h_b_ins[0], b, h, m, n, k);
  transpose(out, b, h, m, n);
  matmul_cpu(out, h_ins[0], h_b_ins[1], b, h, n, m, k);
  // Just revert out back
  transpose(out, b, h, m, n);
  for (int64_t i = 0; i < b * h * m * k; i++) {
    h_b_ins[0][i] = h_b_ins[0][i] / ((float)sqrt(k));
  }
  for (int64_t i = 0; i < b * h * n * k; i++) {
    h_b_ins[1][i] = h_b_ins[1][i] / ((float)sqrt(k));
  }
}

template <typename T>
void multi_head_attention_dgrad_3d_cpu(T *out, T *value_out, T **h_ins, T **h_b_ins,
                                       int64_t batch_size, int64_t head_num, int64_t seq_from,
                                       int64_t seq_to, int64_t hidden_dim) {
  multi_head_attention_dgrad_cpu(out, h_ins, h_b_ins, batch_size, head_num, seq_from, seq_to,
                                 hidden_dim / head_num);
  // h_b_ins[0]: q
  // h_b_ins[1]: k
  // h_b_ins[2]: v
  transpose_qkv_back_cpu(h_b_ins[0], h_b_ins[1], h_b_ins[2], value_out, batch_size, seq_from,
                         seq_to, head_num, hidden_dim);
}
// transpose score,value, return score_dgrad, value_dgrad
template <typename T>
void multi_head_attention_dgrad_cpu_noT(T *out, T **h_ins, T **h_b_ins, int64_t b, int64_t h,
                                        int64_t m, int64_t n, int64_t k) {
  transpose_v_cpu(out, b, m, h, h * k);
  transpose(h_ins[1], b, h, n, k);
  // grad WRT score
  matmul_cpu(out, h_ins[1], h_b_ins[0], b, h, m, k, n);
  transpose(h_ins[0], b, h, m, n);
  // grad WRT value
  matmul_cpu(h_ins[0], out, h_b_ins[1], b, h, n, m, k);
  // transpose back
  transpose(h_ins[1], b, h, k, n);
  transpose(h_ins[0], b, h, n, m);
}
// input : out, h_ins[0]; q_buf, h_ins[1]:k_buf, h_ins[2]:val_buf,
// output: h_b_ins[]
template <typename T>
void multi_head_attention_dgrad_3d_cpu_fused(T *in_dgrad, T **h_ins, T *softmax, T *value_4d,
                                             T **h_b_ins, int64_t batch_size, int64_t head_num,
                                             int64_t seq_from, int64_t seq_to, int64_t hidden_dim,
                                             const std::vector<T> &debug_vector) {
  std::unique_ptr<T[]> tmp_value_dgrad(new T[batch_size * head_num * seq_from * seq_to]);
  T *stage2_input[2] = {softmax, value_4d};
  std::unique_ptr<T[]> softmax_dgrad(new T[batch_size * head_num * seq_from * seq_to]);
  std::unique_ptr<T[]> score_dgrad(new T[batch_size * head_num * seq_from * seq_to]);
  std::unique_ptr<T[]> value_dgrad(new T[batch_size * hidden_dim * seq_to]);
  T *stage2_output[2] = {softmax_dgrad.get(), value_dgrad.get()};

  multi_head_attention_dgrad_cpu_noT(in_dgrad, stage2_input, stage2_output, batch_size, head_num,
                                     seq_from, seq_to, hidden_dim / head_num);
  softmax_bprop_cpu<T>(score_dgrad.get(), softmax_dgrad.get(), softmax,
                       batch_size * head_num * seq_from * seq_to, seq_to);
  h_ins[2] = value_dgrad.get();
  // size_t debug_len = debug_vector.size();
  // for (size_t i = 0; i < debug_len; i++) {
  //   if (score_dgrad[i] - debug_vector[i] > 1e-3) {
  //     std::cout << "error at " << i << " GPU vs CPU: "<<debug_vector[i]<<","<<score_dgrad[i]
  //     <<std::endl;
  //   }
  // }
  // stage2
  // only h_ins[0], h_ins[1] are used
  // h_ins are transposed when fprop()
  multi_head_attention_dgrad_cpu(score_dgrad.get(), h_ins, h_b_ins, batch_size, head_num, seq_from,
                                 seq_to, hidden_dim / head_num);

  // h_b_ins[0]: q
  // h_b_ins[1]: k
  // h_b_ins[2]: v
  transpose_qkv_back_cpu(h_b_ins[0], h_b_ins[1], h_b_ins[2], value_dgrad.get(), batch_size,
                         seq_from, seq_to, head_num, hidden_dim);
}
template <typename T>
void multi_head_attention_layer_test_fused(int64_t batch_size, int64_t seq_from, int64_t seq_to,
                                           int64_t hidden_dim, int head_num,
                                           bool enable_tf32_compute = false) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;
  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());
  std::vector<core23::Tensor> input_3d_tensors;

  int64_t q_size = batch_size * seq_from * hidden_dim;
  int64_t kv_size = batch_size * seq_to * hidden_dim;

  core23::Shape q_shape = {batch_size, seq_from, hidden_dim};
  core23::Shape kv_shape = {batch_size, seq_to, hidden_dim};
  int64_t num = 3;
  input_3d_tensors.emplace_back(tensor_params.shape(q_shape));
  for (int64_t i = 0; i < 2; i++) {
    input_3d_tensors.emplace_back(tensor_params.shape(kv_shape));
  }
  std::vector<core23::Tensor> output_3d_tensors;
  MultiHeadAttentionLayer<T> multi_head_attention_3d_layer(
      input_3d_tensors, output_3d_tensors, head_num, true, test::get_default_gpu(),
      use_mixed_precision, enable_tf32_compute);

  std::unique_ptr<T *[]> h_cpu_ins(new T *[num]);
  std::unique_ptr<T[]> h_d_out(new T[q_size]);
  std::unique_ptr<T[]> tmp_softmax(new T[batch_size * seq_from * seq_to * head_num]);
  std::unique_ptr<T[]> h_cpu_out(new T[q_size]);
  std::unique_ptr<T[]> h_cpu_value_out(new T[kv_size]);
  std::unique_ptr<T *[]> h_d_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  // test 3d input for mha layer
  for (int64_t i = 0; i < num; i++) {
    int64_t in_size = kv_size;
    if (i == 0) {
      in_size = q_size;
    }
    h_cpu_ins[i] = new T[in_size];
    h_d_bprop_out[i] = new T[in_size];
    h_cpu_bprop_out[i] = new T[in_size];

    test::normal_sync_cpu(h_cpu_ins[i], in_size, 0.f, 1.f, generator);
    core23::copy_sync(input_3d_tensors[i].data(), h_cpu_ins[i], input_3d_tensors[i].num_bytes(),
                      input_3d_tensors[i].device(), core23::DeviceType::CPU);
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_3d_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_d_out.get(), output_3d_tensors[0].data(), output_3d_tensors[0].num_bytes(),
                    core23::DeviceType::CPU, output_3d_tensors[0].device());
  // q, k, v will be transposed after fprop()
  multi_head_attention_3d_cpu_fused(h_cpu_ins[0], h_cpu_ins[1], h_cpu_ins[2], h_cpu_out.get(),
                                    tmp_softmax.get(), h_cpu_value_out.get(), batch_size, seq_from,
                                    seq_to, hidden_dim, head_num);
  ASSERT_TRUE(test::compare_array_approx<T>(h_d_out.get(), h_cpu_out.get(), q_size,
                                            get_eps<T>(enable_tf32_compute)));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_3d_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  std::vector<T> &debug_vector = multi_head_attention_3d_layer.get_debug_vector();
  // attention_out: h_d_out as the dgrad
  // output dgrad h_cpu_bprop_out
  multi_head_attention_dgrad_3d_cpu_fused(h_d_out.get(), h_cpu_ins.get(), tmp_softmax.get(),
                                          h_cpu_value_out.get(), h_cpu_bprop_out.get(), batch_size,
                                          head_num, seq_from, seq_to, hidden_dim, debug_vector);
  for (int64_t i = 0; i < num; i++) {
    int64_t in_size = input_3d_tensors[i].num_elements();
    core23::copy_sync(h_d_bprop_out[i], input_3d_tensors[i].data(), input_3d_tensors[i].num_bytes(),
                      core23::DeviceType::CPU, input_3d_tensors[i].device());
    ASSERT_TRUE(test::compare_array_approx<T>(h_d_bprop_out[i], h_cpu_bprop_out[i], in_size,
                                              get_eps<T>(enable_tf32_compute)));
  }
}
}  // namespace
// batch seqlen_from seqlen_to hidden_dim num_head
TEST(mha_layer, fp32_512x30x128) {
  multi_head_attention_layer_test_fused<float>(512, 30, 30, 128, 4);
}

TEST(mha_layer, fp32_asymetric) {
  multi_head_attention_layer_test_fused<float>(512, 30, 60, 128, 4);
  multi_head_attention_layer_test_fused<float>(255, 20, 10, 128, 16);
}
TEST(mha_layer, tf32_256x10x1024) {
  multi_head_attention_layer_test_fused<float>(256, 10, 10, 1024, 8, true);
}
TEST(mha_layer, fp16_debug) {
  multi_head_attention_layer_test_fused<__half>(128, 50, 20, 256, 16);
  multi_head_attention_layer_test_fused<__half>(127, 20, 50, 128, 16);
  multi_head_attention_layer_test_fused<__half>(2, 4, 4, 16, 1);
}

TEST(mha_layer, fp32_debug) { multi_head_attention_layer_test_fused<float>(2, 4, 4, 16, 1); }

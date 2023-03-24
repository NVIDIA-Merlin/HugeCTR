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
void transpose_qkv_cpu(T *query, T *key, T *value, T *value_4d, int64_t batch_size, int64_t seq_len,
                       int64_t head_num, int64_t hidden_dim) {
  auto size = seq_len * hidden_dim;
  auto size_per_head = hidden_dim / head_num;
  std::unique_ptr<T[]> tmp_q(new T[size]);
  std::unique_ptr<T[]> tmp_k(new T[size]);
  for (int64_t b = 0; b < batch_size; b++) {
    T *cur_q = query + b * seq_len * hidden_dim;
    T *cur_k = key + b * seq_len * hidden_dim;
    T *cur_v = value + b * seq_len * hidden_dim;
    T *cur_v_4d = value_4d + b * seq_len * hidden_dim;
    for (int64_t s = 0; s < seq_len; s++) {
      for (int64_t d = 0; d < hidden_dim; d++) {
        tmp_q[(d / size_per_head) * seq_len * size_per_head + s * size_per_head +
              d % size_per_head] = cur_q[s * hidden_dim + d];
        tmp_k[(d / size_per_head) * seq_len * size_per_head + s * size_per_head +
              d % size_per_head] = cur_k[s * hidden_dim + d];
        cur_v_4d[(d / size_per_head) * seq_len * size_per_head + s * size_per_head +
                 d % size_per_head] = cur_v[s * hidden_dim + d];
      }
    }
    for (int64_t i = 0; i < size; ++i) {
      cur_q[i] = tmp_q[i];
      cur_k[i] = tmp_k[i];
    }
  }
}

// transpose from [batch_size, head_num, seq_len, size_per_head] -> [batch_size, seq_len,
// hidden_num]
template <typename T>
void transpose_qkv_back_cpu(T *query, T *key, T *value, T *value_4d, int64_t batch_size,
                            int64_t seq_len, int64_t head_num, int64_t hidden_dim) {
  auto size = seq_len * hidden_dim;
  auto size_per_head = hidden_dim / head_num;
  std::unique_ptr<T[]> tmp_q(new T[size]);
  std::unique_ptr<T[]> tmp_k(new T[size]);
  for (int64_t b = 0; b < batch_size; b++) {
    T *cur_q = query + b * seq_len * hidden_dim;
    T *cur_k = key + b * seq_len * hidden_dim;
    T *cur_v = value + b * seq_len * hidden_dim;
    T *cur_v_4d = value_4d + b * seq_len * hidden_dim;
    for (int64_t h = 0; h < head_num; h++) {
      for (int64_t s = 0; s < seq_len; s++) {
        for (int64_t d = 0; d < size_per_head; d++) {
          tmp_q[s * hidden_dim + h * size_per_head + d] =
              cur_q[h * seq_len * size_per_head + s * size_per_head + d];
          tmp_k[s * hidden_dim + h * size_per_head + d] =
              cur_k[h * seq_len * size_per_head + s * size_per_head + d];
          cur_v[s * hidden_dim + h * size_per_head + d] =
              cur_v_4d[h * seq_len * size_per_head + s * size_per_head + d];
        }
      }
    }
    for (int64_t i = 0; i < size; ++i) {
      cur_q[i] = tmp_q[i];
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

template <typename T>
void multi_head_attention_cpu_noT(T *in1, T *in2, T *output, int64_t b, int64_t h, int64_t m,
                                  int64_t n, int64_t k) {
  matmul_cpu(in1, in2, output, b, h, m, n, k);
  transpose_v_back_cpu(output, b, m, h, h * k);
}

template <typename T>
void multi_head_attention_3d_cpu(T *in1, T *in2, T *in3, T *output, T *value_out,
                                 int64_t batch_size, int64_t seq_len, int64_t hidden_dim,
                                 int64_t head_num) {
  transpose_qkv_cpu(in1, in2, in3, value_out, batch_size, seq_len, head_num, hidden_dim);
  multi_head_attention_cpu(in1, in2, output, batch_size, head_num, seq_len, seq_len,
                           hidden_dim / head_num);
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
template <typename T>
void multi_head_attention_dgrad_3d_cpu(T *out, T *value_out, T **h_ins, T **h_b_ins,
                                       int64_t batch_size, int64_t head_num, int64_t seq_len,
                                       int64_t hidden_dim) {
  multi_head_attention_dgrad_cpu(out, h_ins, h_b_ins, batch_size, head_num, seq_len, seq_len,
                                 hidden_dim / head_num);
  transpose_qkv_back_cpu(h_b_ins[0], h_b_ins[1], h_b_ins[2], value_out, batch_size, seq_len,
                         head_num, hidden_dim);
}

template <typename T>
void multi_head_attention_dgrad_cpu_noT(T *out, T **h_ins, T **h_b_ins, int64_t b, int64_t h,
                                        int64_t m, int64_t n, int64_t k) {
  transpose_v_cpu(out, b, m, h, h * k);
  transpose(h_ins[1], b, h, n, k);
  matmul_cpu(out, h_ins[1], h_b_ins[0], b, h, m, k, n);
  transpose(h_ins[0], b, h, m, n);
  matmul_cpu(h_ins[0], out, h_b_ins[1], b, h, n, m, k);
}
template <typename T>
void multi_head_attention_layer_test_4d(int64_t batch_size, int head_num, int64_t from_seq_len,
                                        int64_t to_seq_len, int64_t size_per_head,
                                        bool enable_tf32_compute = false) {
  bool use_mixed_precision = std::is_same_v<T, __half>;
  int64_t out_size = batch_size * head_num * from_seq_len * to_seq_len;
  int64_t dims = 4;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  std::vector<core23::Tensor> input_tensors;
  core23::Shape in_shape_1 = {batch_size, head_num, from_seq_len, size_per_head};
  input_tensors.emplace_back(tensor_params.shape(in_shape_1));

  core23::Shape in_shape_2 = {batch_size, head_num, to_seq_len, size_per_head};
  input_tensors.emplace_back(tensor_params.shape(in_shape_2));

  std::vector<core23::Tensor> output_tensors;

  MultiHeadAttentionLayer<T> multi_head_attention_layer(input_tensors, output_tensors, head_num,
                                                        true, test::get_default_gpu(),
                                                        use_mixed_precision, enable_tf32_compute);

  int64_t num = 2;
  // std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_ins(new T *[num]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T[]> h_d_out(new T[out_size]);
  std::unique_ptr<T *[]> h_d_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  // test 4d input for mha layer
  for (int64_t i = 0; i < num; i++) {
    int64_t size = batch_size * head_num * input_tensors[i].shape()[dims - 2] *
                   input_tensors[i].shape()[dims - 1];
    h_cpu_ins[i] = new T[size];
    h_d_bprop_out[i] = new T[size];
    h_cpu_bprop_out[i] = new T[size];
    test::normal_sync_cpu(h_cpu_ins[i], size, 0.f, 1.f, generator);
    core23::copy_sync(input_tensors[i].data(), h_cpu_ins[i], input_tensors[i].num_bytes(),
                      input_tensors[i].device(), core23::DeviceType::CPU);
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::shared_ptr<GPUResource> gpu_resource = test::get_default_gpu();
  cudaStreamSynchronize(gpu_resource->get_stream());
  std::cout << cudaGetLastError() << std::endl;

  core23::copy_sync(h_d_out.get(), output_tensors[0].data(), output_tensors[0].num_bytes(),
                    core23::DeviceType::CPU, output_tensors[0].device());
  multi_head_attention_cpu(h_cpu_ins[0], h_cpu_ins[1], h_cpu_out.get(), batch_size, head_num,
                           from_seq_len, to_seq_len, size_per_head);

  ASSERT_TRUE(test::compare_array_approx<T>(h_d_out.get(), h_cpu_out.get(), out_size,
                                            get_eps<T>(enable_tf32_compute)));

  // device bprop
  core23::copy_sync(output_tensors[0].data(), h_cpu_out.get(), output_tensors[0].num_bytes(),
                    output_tensors[0].device(), core23::DeviceType::CPU);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  multi_head_attention_dgrad_cpu(h_cpu_out.get(), h_cpu_ins.get(), h_cpu_bprop_out.get(),
                                 batch_size, head_num, from_seq_len, to_seq_len, size_per_head);
  for (int64_t i = 0; i < num; i++) {
    int64_t size = batch_size * head_num * input_tensors[i].shape()[dims - 2] *
                   input_tensors[i].shape()[dims - 1];
    core23::copy_sync(h_d_bprop_out[i], input_tensors[i].data(), input_tensors[i].num_bytes(),
                      core23::DeviceType::CPU, input_tensors[i].device());
    ASSERT_TRUE(test::compare_array_approx<T>(h_d_bprop_out[i], h_cpu_bprop_out[i], size,
                                              get_eps<T>(enable_tf32_compute)));  // compare dgrad
  }
}
template <typename T>
void multi_head_attention_layer_test_4d_noT(int64_t batch_size, int head_num, int64_t from_seq_len,
                                            int64_t to_seq_len, int64_t size_per_head,
                                            bool enable_tf32_compute = false) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;
  int64_t out_size = batch_size * head_num * from_seq_len * size_per_head;
  int64_t dims = 4;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  std::vector<core23::Tensor> input_tensors;
  core23::Shape in_shape_1 = {batch_size, head_num, from_seq_len, to_seq_len};
  input_tensors.emplace_back(tensor_params.shape(in_shape_1));

  core23::Shape in_shape_2 = {batch_size, head_num, to_seq_len, size_per_head};
  input_tensors.emplace_back(tensor_params.shape(in_shape_2));

  std::vector<core23::Tensor> output_tensors;

  MultiHeadAttentionLayer<T> multi_head_attention_layer(input_tensors, output_tensors, head_num,
                                                        false, test::get_default_gpu(),
                                                        use_mixed_precision, enable_tf32_compute);

  int64_t num = 2;

  std::unique_ptr<T *[]> h_cpu_ins(new T *[num]);
  std::unique_ptr<T[]> h_d_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T *[]> h_d_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  // test 4d input for mha layer
  for (int64_t i = 0; i < num; i++) {
    int64_t size = batch_size * head_num * input_tensors[i].shape()[dims - 2] *
                   input_tensors[i].shape()[dims - 1];
    h_cpu_ins[i] = new T[size];
    h_d_bprop_out[i] = new T[size];
    h_cpu_bprop_out[i] = new T[size];
    test::normal_sync_cpu(h_cpu_ins[i], size, 0.f, 1.f, generator);
    core23::copy_sync(input_tensors[i].data(), h_cpu_ins[i], input_tensors[i].num_bytes(),
                      input_tensors[i].device(), core23::DeviceType::CPU);
  }
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::shared_ptr<GPUResource> gpu_resource = test::get_default_gpu();
  cudaStreamSynchronize(gpu_resource->get_stream());
  std::cout << cudaGetLastError() << std::endl;

  core23::copy_sync(h_d_out.get(), output_tensors[0].data(), output_tensors[0].num_bytes(),
                    core23::DeviceType::CPU, output_tensors[0].device());
  multi_head_attention_cpu_noT(h_cpu_ins[0], h_cpu_ins[1], h_cpu_out.get(), batch_size, head_num,
                               from_seq_len, to_seq_len, size_per_head);

  ASSERT_TRUE(test::compare_array_approx<T>(h_d_out.get(), h_cpu_out.get(), out_size,
                                            get_eps<T>(enable_tf32_compute)));

  // device bprop
  core23::copy_sync(output_tensors[0].data(), h_cpu_out.get(), output_tensors[0].num_bytes(),
                    output_tensors[0].device(), core23::DeviceType::CPU);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  cudaStreamSynchronize(gpu_resource->get_stream());
  std::cout << cudaGetLastError() << std::endl;

  multi_head_attention_dgrad_cpu_noT(h_cpu_out.get(), h_cpu_ins.get(), h_cpu_bprop_out.get(),
                                     batch_size, head_num, from_seq_len, to_seq_len, size_per_head);
  for (int64_t i = 0; i < num; i++) {
    int64_t size = batch_size * head_num * input_tensors[i].shape()[dims - 2] *
                   input_tensors[i].shape()[dims - 1];
    // HCTR_LIB_THROW(cudaMemcpy(h_bprop_out[i], d_out, size * sizeof(T), cudaMemcpyDeviceToHost));
    core23::copy_sync(h_d_bprop_out[i], input_tensors[i].data(), input_tensors[i].num_bytes(),
                      core23::DeviceType::CPU, input_tensors[i].device());
    ASSERT_TRUE(test::compare_array_approx<T>(h_d_bprop_out[i], h_cpu_bprop_out[i], size,
                                              get_eps<T>(enable_tf32_compute)));  // compare dgrad
  }
}
template <typename T>
void multi_head_attention_layer_test_3d(int64_t batch_size, int64_t seq_len, int64_t hidden_dim,
                                        int head_num, bool enable_tf32_compute = false) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;
  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());
  std::vector<core23::Tensor> input_3d_tensors;

  int64_t in_size = batch_size * seq_len * hidden_dim;
  int64_t out_size = batch_size * head_num * seq_len * seq_len;

  core23::Shape in_shape = {batch_size, seq_len, hidden_dim};
  int64_t num = 3;
  for (int64_t i = 0; i < num; i++) {
    input_3d_tensors.emplace_back(tensor_params.shape(in_shape));
  }
  std::vector<core23::Tensor> output_3d_tensors;

  MultiHeadAttentionLayer<T> multi_head_attention_3d_layer(
      input_3d_tensors, output_3d_tensors, head_num, true, test::get_default_gpu(),
      use_mixed_precision, enable_tf32_compute);

  std::unique_ptr<T *[]> h_cpu_ins(new T *[num]);
  std::unique_ptr<T[]> h_d_out(new T[out_size]);
  std::unique_ptr<T[]> h_d_value_out(new T[in_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_value_out(new T[in_size]);
  std::unique_ptr<T *[]> h_d_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  // test 3d input for mha layer
  for (int64_t i = 0; i < num; i++) {
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
  core23::copy_sync(h_d_value_out.get(), output_3d_tensors[1].data(),
                    output_3d_tensors[1].num_bytes(), core23::DeviceType::CPU,
                    output_3d_tensors[1].device());
  multi_head_attention_3d_cpu(h_cpu_ins[0], h_cpu_ins[1], h_cpu_ins[2], h_cpu_out.get(),
                              h_cpu_value_out.get(), batch_size, seq_len, hidden_dim, head_num);
  ASSERT_TRUE(test::compare_array_approx<T>(h_d_out.get(), h_cpu_out.get(), out_size,
                                            get_eps<T>(enable_tf32_compute)));
  ASSERT_TRUE(test::compare_array_approx<T>(h_d_value_out.get(), h_cpu_value_out.get(), in_size,
                                            get_eps<T>(enable_tf32_compute)));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_3d_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  multi_head_attention_dgrad_3d_cpu(h_d_out.get(), h_d_value_out.get(), h_cpu_ins.get(),
                                    h_cpu_bprop_out.get(), batch_size, head_num, seq_len,
                                    hidden_dim);
  for (int64_t i = 0; i < num; i++) {
    core23::copy_sync(h_d_bprop_out[i], input_3d_tensors[i].data(), input_3d_tensors[i].num_bytes(),
                      core23::DeviceType::CPU, input_3d_tensors[i].device());
    ASSERT_TRUE(test::compare_array_approx<T>(h_d_bprop_out[i], h_cpu_bprop_out[i], in_size,
                                              get_eps<T>(enable_tf32_compute)));  // compare dgrad
  }
}
}  // namespace

TEST(mha_layer, fp32_512x4x400x600) {
  multi_head_attention_layer_test_4d<float>(512, 4, 400, 600, 128);
}

TEST(mha_layer, tf32_512x4x200x200) {
  multi_head_attention_layer_test_4d<float>(512, 4, 200, 200, 256, true);
}

TEST(mha_layer, fp16_512x4x100x200) {
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  if (major < 7) {
    GTEST_SKIP();
  }
  multi_head_attention_layer_test_4d<__half>(512, 4, 100, 200, 256, false);
}

TEST(mha_layer, fp32_512x4x400x128) {
  multi_head_attention_layer_test_4d_noT<float>(512, 4, 400, 200, 128);
}

TEST(mha_layer, fp16_512x4x400x128_test) {
  multi_head_attention_layer_test_4d_noT<float>(512, 4, 200, 100, 64);
}
TEST(mha_layer, tf32_512x4x200x256) {
  multi_head_attention_layer_test_4d_noT<float>(512, 4, 200, 200, 256, true);
}

TEST(mha_layer, fp32_512x300x128) { multi_head_attention_layer_test_3d<float>(512, 300, 128, 4); }
TEST(mha_layer, tf32_256x100x1024) {
  multi_head_attention_layer_test_3d<float>(256, 100, 1024, 8, true);
}
TEST(mha_layer, fp16_128x200x256) { multi_head_attention_layer_test_3d<__half>(128, 200, 256, 16); }

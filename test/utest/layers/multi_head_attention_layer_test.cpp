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

#include "HugeCTR/include/layers/multi_head_attention_layer.hpp"

#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <math.h>
#include <utest/test_utils.h>

#include <memory>
#include <utils.hpp>
#include <vector>

#include "HugeCTR/include/utils.hpp"
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
void matmul_cpu(T *in1, T *in2, T *output, size_t H, size_t B, size_t M, size_t N, size_t K) {
  //(m,n)x(n,k)
  size_t i, j, k, z, y;
  for (y = 0; y < H; y++) {
    for (z = 0; z < B; z++) {
      for (i = 0; i < M; i++) {
        for (j = 0; j < K; j++) {
          output[y * B * M * K + z * M * K + i * K + j] = TypeConvert<T, float>::convert(0.0f);
          for (k = 0; k < N; k++) {
            output[y * B * M * K + z * M * K + i * K + j] =
                output[y * B * M * K + z * M * K + i * K + j] +
                in1[y * B * M * N + z * M * N + i * N + k] *
                    in2[y * B * N * K + z * N * K + k * K + j];
          }
        }
      }
    }
  }
}

template <typename T>
static void transpose(T *a, size_t h, size_t b, size_t m, size_t n) {
  for (size_t y = 0; y < h; y++) {
    for (size_t z = 0; z < b; z++) {
      T *cur_a = a + z * m * n + y * b * m * n;
      std::unique_ptr<T[]> tmp(new T[m * n]);
      for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) tmp[j * m + i] = cur_a[i * n + j];
      for (size_t i = 0; i < m * n; ++i) cur_a[i] = tmp[i];
    }
  }
}

template <typename T>
void multi_head_attention_cpu(T *in1, T *in2, T *output, size_t h, size_t b, size_t m, size_t n,
                              size_t k) {
  transpose(in2, h, b, n, k);
  matmul_cpu(in1, in2, output, h, b, m, k, n);
  // Just to revert in2 back
  transpose(in2, h, b, k, n);
}

template <typename T>
void multi_head_attention_dgrad_cpu(T *out, T **h_ins, T **h_b_ins, size_t h, size_t b, size_t m,
                                    size_t n, size_t k) {
  // transpose(h_ins[1], h, b, n, k);
  // transpose(h_ins[0], h, b, m, n);
  // out [h,b,m,n]
  // in1 [h,b,m,k]
  // in2 [h,b,n,k]
  matmul_cpu(out, h_ins[1], h_b_ins[0], h, b, m, n, k);
  transpose(out, h, b, m, n);
  matmul_cpu(out, h_ins[0], h_b_ins[1], h, b, n, m, k);
  // Just revert out back
  transpose(out, h, b, m, n);
}

template <typename T>
void multi_head_attention_layer_test(size_t head_num, size_t batch_size, size_t from_seq_len,
                                     size_t to_seq_len, size_t size_per_head,
                                     bool use_mixed_precision = false,
                                     bool enable_tf32_compute = false) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  size_t out_size = head_num * batch_size * from_seq_len * to_seq_len;
  int dims = 4;

  Tensors2<T> in_tensors;
  Tensor2<T> in_tensor;

  buff->reserve({head_num, batch_size, from_seq_len, size_per_head}, &in_tensor);
  in_tensors.push_back(in_tensor);
  buff->reserve({head_num, batch_size, to_seq_len, size_per_head}, &in_tensor);
  in_tensors.push_back(in_tensor);
  Tensor2<T> out_tensor;
  buff->reserve({head_num, batch_size, from_seq_len, to_seq_len}, &out_tensor);

  MultiHeadAttentionLayer multi_head_attention_layer(in_tensors, out_tensor, buff,
                                                     test::get_default_gpu(), use_mixed_precision,
                                                     enable_tf32_compute);
  buff->allocate();

  size_t num = 2;
  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T *[]> h_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // device fprop
  for (size_t i = 0; i < num; i++) {
    size_t size = head_num * batch_size * in_tensors[i].get_dimensions()[dims - 2] *
                  in_tensors[i].get_dimensions()[dims - 1];
    h_ins[i] = new T[size];
    h_bprop_out[i] = new T[size];
    h_cpu_bprop_out[i] = new T[size];
    simulator.fill(h_ins[i], test::align_to_even(size));
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }
  /*cout << "Before fprop" << endl;
  for (size_t i = 0; i < num; i++) {
    cout << "Input " << i << endl;
    for (size_t j = 0; j < head_num * batch_size * in_tensors[i].get_dimensions()[dims - 2] *
                               in_tensors[i].get_dimensions()[dims - 1];
         j++) {
      cout << h_ins[i][j] << " ";
      if (((j + 1) % in_tensors[i].get_dimensions()[dims - 1]) == 0) {
        cout << endl;
        if (((j + 1) % (in_tensors[i].get_dimensions()[dims - 2] *
                        in_tensors[i].get_dimensions()[dims - 1])) == 0) {
          cout << endl;
        }
      }
    }
  }*/

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  T *d_out = out_tensor.get_ptr();
  HCTR_LIB_THROW(
      cudaMemcpy(h_out.get(), d_out, out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToHost));
  multi_head_attention_cpu(h_ins[0], h_ins[1], h_cpu_out.get(), head_num, batch_size, from_seq_len,
                           to_seq_len, size_per_head);

  /*cout << "After fprop" << endl;
  cout << "Output " << endl;
  for (size_t j = 0; j < out_size; j++) {
    cout << h_out.get()[j] << " ";
    if (((j + 1) % to_seq_len) == 0) {
      cout << endl;
      if (((j + 1) % (from_seq_len * to_seq_len)) == 0) {
        cout << endl;
      }
    }
  }

  cout << "Cpu Output " << endl;
  for (size_t j = 0; j < out_size; j++) {
    cout << h_cpu_out.get()[j] << " ";
    if (((j + 1) % to_seq_len) == 0) {
      cout << endl;
      if (((j + 1) % (from_seq_len * to_seq_len)) == 0) {
        cout << endl;
      }
    }
  }*/

  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), out_size,
                                            get_eps<T>(enable_tf32_compute)));

  // device bprop
  simulator.fill(h_out.get(), test::align_to_even(out_size));
  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), out_size * sizeof(T), cudaMemcpyHostToDevice));

  /*cout << "Before bprop  " << endl;
  cout << "Output " << endl;
  for (size_t j = 0; j < out_size; j++) {
    cout << h_out.get()[j] << " ";
    if (((j + 1) % to_seq_len) == 0) {
      cout << endl;
      if (((j + 1) % (from_seq_len * to_seq_len)) == 0) {
        cout << endl;
      }
    }
  }
  cout << "Cpu Output " << endl;
  for (size_t j = 0; j < out_size; j++) {
    cout << h_out.get()[j] << " ";
    if (((j + 1) % to_seq_len) == 0) {
      cout << endl;
      if (((j + 1) % (from_seq_len * to_seq_len)) == 0) {
        cout << endl;
      }
    }
  }
  cout << "Device Input " << endl;
  for (size_t i = 0; i < num; i++) {
    cout << "Input " << i << endl;
    for (size_t j = 0;
         j <  head_num * batch_size * in_tensors[i].get_dimensions()[dims - 2] *
                 in_tensors[i].get_dimensions()[dims - 1];
         j++) {
      cout << h_d_ins[i][j] << " ";
      if (((j + 1) % in_tensors[i].get_dimensions()[dims - 1]) == 0) {
        cout << endl;
        if (((j + 1) % (in_tensors[i].get_dimensions()[dims - 2] *
                        in_tensors[i].get_dimensions()[dims - 1])) == 0) {
          cout << endl;
        }
      }
    }
  }
  cout << "CPU Input " << endl;
  for (size_t i = 0; i < num; i++) {
    cout << "Input " << i << endl;
    for (size_t j = 0;
         j < batch_size * head_num * batch_size * in_tensors[i].get_dimensions()[dims - 2] *
                 in_tensors[i].get_dimensions()[dims - 1];
         j++) {
      cout << h_ins[i][j] << " ";
      if (((j + 1) % in_tensors[i].get_dimensions()[dims - 1]) == 0) {
        cout << endl;
        if (((j + 1) % (in_tensors[i].get_dimensions()[dims - 2] *
                        in_tensors[i].get_dimensions()[dims - 1])) == 0) {
          cout << endl;
        }
      }
    }
  }*/

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  multi_head_attention_dgrad_cpu(h_out.get(), h_ins.get(), h_cpu_bprop_out.get(), head_num,
                                 batch_size, from_seq_len, to_seq_len, size_per_head);
  for (size_t i = 0; i < num; i++) {
    size_t size = head_num * batch_size * in_tensors[i].get_dimensions()[dims - 2] *
                  in_tensors[i].get_dimensions()[dims - 1];
    // cout << "Input bprop  " << i << " Size " << size << endl;
    T *d_out = in_tensors[i].get_ptr();
    HCTR_LIB_THROW(cudaMemcpy(h_bprop_out[i], d_out, size * sizeof(T), cudaMemcpyDeviceToHost));
    /*cout << "GPU result " << endl;
    for (size_t j = 0; j < size; j++) {
      cout << h_bprop_out[i][j] << " ";
      if (((j + 1) % in_tensors[i].get_dimensions()[dims - 1]) == 0) {
        cout << endl;
        if (((j + 1) % (in_tensors[i].get_dimensions()[dims - 2] *
                        in_tensors[i].get_dimensions()[dims - 1])) == 0) {
          cout << endl;
        }
      }
    }
    cout << "CPU result " << endl;
    for (size_t j = 0; j < size; j++) {
      cout << h_cpu_bprop_out[i][j] << " ";
      if (((j + 1) % in_tensors[i].get_dimensions()[dims - 1]) == 0) {
        cout << endl;
        if (((j + 1) % (in_tensors[i].get_dimensions()[dims - 2] *
                        in_tensors[i].get_dimensions()[dims - 1])) == 0) {
          cout << endl;
        }
      }
    }*/
    ASSERT_TRUE(test::compare_array_approx<T>(h_bprop_out[i], h_cpu_bprop_out[i], size,
                                              get_eps<T>(enable_tf32_compute)));  // compare dgrad
  }
}

}  // namespace

TEST(mha_layer, fp32_512x512) { multi_head_attention_layer_test<float>(4, 512, 400, 600, 128); }

TEST(mha_layer, tf32_512x1024) {
  multi_head_attention_layer_test<float>(4, 512, 200, 200, 256, false, true);
}

TEST(mha_layer, fp16_512x479) {
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  if (major < 7) {
    GTEST_SKIP();
  }
  multi_head_attention_layer_test<__half>(4, 512, 100, 200, 256, true, false);
}

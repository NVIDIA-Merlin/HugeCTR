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

#include "HugeCTR/include/layers/interaction_layer.hpp"

#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <math.h>
#include <utest/test_utils.h>

#include <memory>
#include <utils.hpp>
#include <vector>

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
void interaction_layer_test(size_t height, size_t n_emb, size_t in_width,
                            bool enable_tf32_compute = false) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  Tensors2<T> in_tensors;

  test::GaussianDataSimulator data_sim(0.0f, 1.0f);
  std::vector<std::vector<T>> h_ins;
  for (int ni = 0; ni < 2; ni++) {
    std::vector<size_t> dims;
    if (ni == 0) {
      dims = {height, in_width};
    } else {
      dims = {height, n_emb, in_width};
    }
    Tensor2<T> in_tensor;
    buff->reserve(dims, &in_tensor);
    in_tensors.push_back(in_tensor);

    h_ins.push_back(
        std::vector<T>(in_tensor.get_num_elements(), TypeConvert<T, float>::convert(0.0f)));

    data_sim.fill(h_ins[ni].data(), h_ins[ni].size());
  }

  auto& in_mlp_tensor = in_tensors[0];
  auto& in_emb_tensor = in_tensors[1];

  auto& h_in_mlp = h_ins[0];
  auto& h_in_emb = h_ins[1];

  size_t n_ins = 1 + n_emb;
  size_t out_width = n_ins * in_width;

  std::vector<T> h_concat(height * out_width, TypeConvert<T, float>::convert(0.0f));
  auto concat_op = [&](bool fprop) {
    for (size_t ni = 0; ni < n_ins; ni++) {
      for (size_t h = 0; h < height; h++) {
        size_t in_idx_base = (ni == 0) ? h * in_width : h * in_width * n_emb;
        for (size_t w = 0; w < in_width; w++) {
          size_t in_idx = in_idx_base + w;
          size_t out_idx = h * out_width + ni * in_width + w;
          if (fprop) {
            h_concat[out_idx] =
                (ni == 0) ? h_in_mlp[in_idx] : h_in_emb[(ni - 1) * in_width + in_idx];
          } else {
            if (ni == 0) {
              h_in_mlp[in_idx] = h_in_mlp[in_idx] + h_concat[out_idx];
            } else {
              h_in_emb[in_idx + (ni - 1) * in_width] = h_concat[out_idx];
            }
          }
        }
      }
    }
  };

  Tensor2<T> out_tensor;
  InteractionLayer<T> interaction_layer(in_mlp_tensor, in_emb_tensor, out_tensor, buff,
                                        test::get_default_gpu(), false, enable_tf32_compute);

  buff->allocate();
  interaction_layer.initialize();

  // device fprop
  T* d_in_mlp = in_mlp_tensor.get_ptr();
  HCTR_LIB_THROW(cudaMemcpy(d_in_mlp, &h_in_mlp.front(), in_mlp_tensor.get_size_in_bytes(),
                            cudaMemcpyHostToDevice));
  T* d_in_emb = in_emb_tensor.get_ptr();
  HCTR_LIB_THROW(cudaMemcpy(d_in_emb, &h_in_emb.front(), in_emb_tensor.get_size_in_bytes(),
                            cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  interaction_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  // host fprop
  concat_op(true);

  std::vector<T> h_mat(height * n_ins * n_ins, TypeConvert<T, float>::convert(0.0f));
  for (size_t p = 0; p < height; p++) {
    size_t concat_stride = n_ins * in_width * p;
    size_t mat_stride = n_ins * n_ins * p;
    for (size_t m = 0; m < n_ins; m++) {
      for (size_t n = 0; n < n_ins; n++) {
        float accum = 0.0f;
        for (size_t k = 0; k < in_width; k++) {
          accum += h_concat[concat_stride + m * in_width + k] *
                   h_concat[concat_stride + n * in_width + k];
        }
        h_mat[mat_stride + m * n_ins + n] = accum;
      }
    }
  }

  size_t out_len = in_width + (n_ins * (n_ins + 1) / 2 - n_ins) + 1;
  std::vector<T> h_ref(height * out_len, 0.0);
  for (size_t p = 0; p < height; p++) {
    size_t cur_idx = 0;
    size_t out_stride = p * out_len;
    size_t mat_stride = p * n_ins * n_ins;
    for (size_t i = 0; i < in_width; i++) {
      h_ref[out_stride + cur_idx++] = h_in_mlp[p * in_width + i];
    }
    for (size_t n = 0; n < n_ins; n++) {
      for (size_t m = 0; m < n_ins; m++) {
        if (n > m) {
          h_ref[out_stride + cur_idx++] = h_mat[mat_stride + m * n_ins + n];
        }
      }
    }
  }

  std::vector<T> h_out(out_tensor.get_num_elements(), TypeConvert<T, float>::convert(0.0f));
  T* d_out = out_tensor.get_ptr();
  HCTR_LIB_THROW(
      cudaMemcpy(&h_out.front(), d_out, out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToHost));

  ASSERT_TRUE(test::compare_array_approx<T>(&h_out.front(), &h_ref.front(), h_out.size(),
                                            get_eps<T>(enable_tf32_compute)));

  // device bprop
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  interaction_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  // host bprop
  for (size_t p = 0; p < height; p++) {
    size_t cur_idx = 0;
    size_t out_stride = p * out_len;
    size_t mat_stride = p * n_ins * n_ins;
    for (size_t i = 0; i < in_width; i++) {
      h_in_mlp[p * in_width + i] = h_ref[out_stride + cur_idx++];
    }
    for (size_t n = 0; n < n_ins; n++) {
      for (size_t m = 0; m < n_ins; m++) {
        h_mat[mat_stride + m * n_ins + n] =
            (n > m) ? h_ref[out_stride + cur_idx++] : TypeConvert<T, float>::convert(0.0f);
      }
    }
  }

  std::vector<T> h_concat_tmp(h_concat);
  for (size_t p = 0; p < height; p++) {
    size_t mat_stride = n_ins * n_ins * p;
    size_t concat_stride = n_ins * in_width * p;
    for (size_t m = 0; m < n_ins; m++) {
      for (size_t n = 0; n < in_width; n++) {
        float accum = 0.0f;
        for (size_t k = 0; k < n_ins; k++) {
          accum += (h_mat[mat_stride + m * n_ins + k] + h_mat[mat_stride + k * n_ins + m]) *
                   h_concat_tmp[concat_stride + k * in_width + n];
        }
        h_concat[concat_stride + m * in_width + n] = 1.0f * accum;
      }
    }
  }
  concat_op(false);

  for (int i = 0; i < 2; i++) {
    auto in_tensor = in_tensors[i];
    std::vector<T> h_in(in_tensor.get_num_elements(), TypeConvert<T, float>::convert(0.0f));
    T* d_in = in_tensor.get_ptr();
    HCTR_LIB_THROW(
        cudaMemcpy(&h_in.front(), d_in, in_tensor.get_size_in_bytes(), cudaMemcpyDeviceToHost));
    std::vector<T>& h_ref = h_ins[i];

    ASSERT_TRUE(test::compare_array_approx<T>(&h_in.front(), &h_ref.front(), h_in.size(),
                                              get_eps<T>(enable_tf32_compute)));
  }
}

}  // namespace

TEST(interaction_layer, fp32_512x479) { interaction_layer_test<float>(512, 26, 128); }
TEST(interaction_layer, tf32_512x479) { interaction_layer_test<float>(512, 26, 128, true); }

TEST(interaction_layer, fp16_512x479) {
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  if (major < 7) {
    GTEST_SKIP();
  }
  interaction_layer_test<__half>(512, 26, 128);
}

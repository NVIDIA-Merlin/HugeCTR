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

#include "HugeCTR/include/layers/interaction_layer.hpp"

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "cublas_v2.h"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <math.h>
#include <memory>
#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-3;

template <typename T>
void interaction_layer_test(size_t height, size_t n_emb, size_t in_width) {
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  std::shared_ptr<GeneralBuffer<T>> buff(new GeneralBuffer<T>());
  Tensors<T> in_tensors;

  GaussianDataSimulator<float> data_sim(0.0, 1.0, -0.25, 0.25);
  std::vector<std::vector<T>> h_ins;
  for (int ni = 0; ni < 2; ni++) {
    std::vector<size_t> dims;
    if (ni == 0) {
      dims = {height, in_width};
    } else {
      dims = {height, n_emb, in_width};
    }
    TensorFormat_t in_format = (ni == 0) ? TensorFormat_t::HW : TensorFormat_t::HSW;
    std::shared_ptr<Tensor<T>> in_tensor(new Tensor<T>(dims, buff, in_format));
    in_tensors.push_back(in_tensor);

    h_ins.push_back(std::vector<T>(in_tensor->get_num_elements(), TypeConvert<T>::convert(0.0f)));
    for (unsigned int i = 0; i < h_ins[ni].size(); i++) {
      h_ins[ni][i] = TypeConvert<T>::convert(data_sim.get_num());
    }
  }

  auto& in_mlp_tensor = in_tensors[0];
  auto& in_emb_tensor = in_tensors[1];

  auto& h_in_mlp = h_ins[0];
  auto& h_in_emb = h_ins[1];

  size_t n_ins = 1 + n_emb;
  size_t out_width = n_ins * in_width;

  std::vector<T> h_concat(height * out_width, TypeConvert<T>::convert(0.0f));
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

  std::shared_ptr<Tensor<T>> out_tensor;
  InteractionLayer<T> interaction_layer(in_mlp_tensor, in_emb_tensor, out_tensor, buff,
                                        cublas_handle, false, 0);

  buff->init(0);

  // device fprop
  T* d_in_mlp = in_mlp_tensor->get_ptr();
  cudaMemcpy(d_in_mlp, &h_in_mlp.front(), in_mlp_tensor->get_size(), cudaMemcpyHostToDevice);
  T* d_in_emb = in_emb_tensor->get_ptr();
  cudaMemcpy(d_in_emb, &h_in_emb.front(), in_emb_tensor->get_size(), cudaMemcpyHostToDevice);
  interaction_layer.fprop(cudaStreamDefault);

  // host fprop
  concat_op(true);

  std::vector<T> h_mat(height * n_ins * n_ins, TypeConvert<T>::convert(0.0f));
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

  std::vector<T> h_out(out_tensor->get_num_elements(), TypeConvert<T>::convert(0.0f));
  T* d_out = out_tensor->get_ptr();
  cudaMemcpy(&h_out.front(), d_out, out_tensor->get_size(), cudaMemcpyDeviceToHost);

  ASSERT_TRUE(
      test::compare_array_approx<T>(&h_out.front(), &h_ref.front(), h_out.size(), TypeConvert<T>::convert(eps)));

  // device bprop
  interaction_layer.bprop(cudaStreamDefault);

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
            (n > m) ? h_ref[out_stride + cur_idx++] : TypeConvert<T>::convert(0.0f);
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
    std::vector<T> h_in(in_tensor->get_num_elements(), TypeConvert<T>::convert(0.0f));
    T* d_in = in_tensor->get_ptr();
    cudaMemcpy(&h_in.front(), d_in, in_tensor->get_size(), cudaMemcpyDeviceToHost);
    std::vector<T>& h_ref = h_ins[i];

    ASSERT_TRUE(test::compare_array_approx<T>(&h_in.front(), &h_ref.front(), h_in.size(), eps));
  }

  cublasDestroy(cublas_handle);
}

}  // namespace

TEST(interaction_layer, fp32_512x479) { interaction_layer_test<float>(512, 26, 128); }
TEST(interaction_layer, fp16_512x479) { interaction_layer_test<__half>(512, 26, 128); }

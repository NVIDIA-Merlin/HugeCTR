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

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/layers/fully_connected_layer.hpp"
#include "HugeCTR/include/loss.hpp"
#include "HugeCTR/include/regularizers/l1_regularizer.hpp"
#include "HugeCTR/include/regularizers/l2_regularizer.hpp"
#include "HugeCTR/include/regularizers/no_regularizer.hpp"

#include <curand.h>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include "cublas_v2.h"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

float get_ref_term(Regularizer_t type, std::vector<float>& h_weight, float lambda, size_t batch_size) {
  float ref_term = 0.0f;
  switch (type) {
    case Regularizer_t::L1: {
      for (auto& v : h_weight) {
        ref_term += fabs(v);
      }
      const float alpha = lambda / batch_size;
      ref_term *= alpha;
      break;
    }
    case Regularizer_t::L2: {
      for (auto& v : h_weight) {
        ref_term += (v * v);
      }
      const float alpha = lambda / (batch_size * 2);
      ref_term *= alpha;
      break;
    }
    default:
      assert(!"Error: no such Regularizer && should never get here!");
      break;
  }
  return ref_term;
}

void get_ref_grad(Regularizer_t type, const std::vector<float>& h_weight,
                  std::vector<float>& h_wgrad, float lambda, size_t batch_size) {
  switch (type) {
    case Regularizer_t::L1: {
      for (size_t i = 0; i < h_wgrad.size(); i++) {
        float sign = (h_weight[i] > 0.0f) ? 1.0f : -1.0f;
        h_wgrad[i] += (lambda / batch_size) * sign;
      }
      break;
    }
    case Regularizer_t::L2: {
      for (size_t i = 0; i < h_wgrad.size(); i++) {
        h_wgrad[i] += (lambda / batch_size) * h_weight[i];
      }
      break;
    }
    default:
      assert(!"Error: no such Regularizer && should never get here!");
      break;
  }
}

std::shared_ptr<Regularizer> create_regularizer(Regularizer_t type,
                                                std::shared_ptr<GeneralBuffer<float>> weight_buff,
                                                std::shared_ptr<GeneralBuffer<float>> wgrad_buff,
                                                size_t batch_size, float lambda,
                                                cublasHandle_t cublas_handle) {
  std::shared_ptr<Regularizer> reg;
  switch (type) {
    case Regularizer_t::L1:
      reg.reset(new L1Regularizer(weight_buff, wgrad_buff, batch_size, lambda, cublas_handle, 0));
      break;
    case Regularizer_t::L2:
      reg.reset(new L2Regularizer(weight_buff, wgrad_buff, batch_size, lambda, cublas_handle, 0));
      break;
    default:
      assert(!"Error: no such optimizer && should never get here!");
      break;
  }
  return reg;
}

const float eps = 1e-5;

void loss_with_regularizer_test(Regularizer_t type, size_t batch_size, size_t num_features,
                                float lambda) {
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  std::shared_ptr<GeneralBuffer<float>> weight_buff_no(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> weight_buff_re(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> wgrad_buff_no(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> wgrad_buff_re(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> blobs_buff(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> label_buff(new GeneralBuffer<float>());

  std::shared_ptr<Tensor<float>> in_tensor(
      new Tensor<float>({batch_size, num_features}, blobs_buff, TensorFormat_t::HW));

  std::shared_ptr<Tensor<float>> out_tensor(
      new Tensor<float>({batch_size, 1}, blobs_buff, TensorFormat_t::HW));

  FullyConnectedLayer fc_layer_no(weight_buff_no, wgrad_buff_no, in_tensor, out_tensor,
                                  TensorFormat_t::HW, cublas_handle, 0);

  FullyConnectedLayer fc_layer_re(weight_buff_re, wgrad_buff_re, in_tensor, out_tensor,
                                  TensorFormat_t::HW, cublas_handle, 0);

  std::shared_ptr<Tensor<float>> loss_tensor_no(
      new Tensor<float>({1, 1}, blobs_buff, TensorFormat_t::HW));

  std::shared_ptr<Tensor<float>> loss_tensor_re(
      new Tensor<float>({1, 1}, blobs_buff, TensorFormat_t::HW));

  std::shared_ptr<Tensor<float>> label_tensor(
      new Tensor<float>({batch_size, 1}, label_buff, TensorFormat_t::HW));

  BinaryCrossEntropyLoss loss_no(label_tensor, out_tensor, loss_tensor_no,
                                 std::shared_ptr<NoRegularizer>(new NoRegularizer(
                                     weight_buff_no, wgrad_buff_no, batch_size, 0)),
                                 0);

  BinaryCrossEntropyLoss loss_re(
      label_tensor, out_tensor, loss_tensor_re,
      create_regularizer(type, weight_buff_re, wgrad_buff_re, batch_size, lambda, cublas_handle),
      0);

  weight_buff_no->init(0);
  weight_buff_re->init(0);
  wgrad_buff_no->init(0);
  wgrad_buff_re->init(0);
  blobs_buff->init(0);
  label_buff->init(0);

  GaussianDataSimulator<float> input_simulator(0.0, 1.0, -1.0, 1.0);
  std::vector<float> h_input(in_tensor->get_num_elements());
  for (size_t i = 0; i < h_input.size(); i++) {
    h_input[i] = input_simulator.get_num();
  }
  cudaMemcpy(in_tensor->get_ptr(), &h_input.front(), in_tensor->get_size(), cudaMemcpyHostToDevice);

  float sigma = 1.f / sqrt(num_features);
  GaussianDataSimulator<float> weight_simulator(0.0, sigma, -2 * sigma, 2 * sigma);
  std::vector<float> h_weight(weight_buff_re->get_num_elements());
  for (size_t i = 0; i < h_weight.size(); i++) {
    h_weight[i] = weight_simulator.get_num();
  }
  cudaMemcpy(weight_buff_no->get_ptr_with_offset(0), &h_weight.front(), weight_buff_no->get_size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(weight_buff_re->get_ptr_with_offset(0), &h_weight.front(), weight_buff_re->get_size(),
             cudaMemcpyHostToDevice);

  UnifiedDataSimulator<int> label_simulator(0, 1);
  std::vector<float> h_label(label_tensor->get_num_elements());
  for (size_t i = 0; i < h_label.size(); i++) {
    h_label[i] = (float)label_simulator.get_num();
  }
  cudaMemcpy(label_tensor->get_ptr(), &h_label.front(), label_tensor->get_size(),
             cudaMemcpyHostToDevice);

  fc_layer_no.fprop(cudaStreamDefault);
  loss_no.fused_loss_computation(cudaStreamDefault);
  std::unique_ptr<float> loss_no_val(new float);
  cudaMemcpy(loss_no_val.get(), loss_tensor_no->get_ptr(), loss_tensor_no->get_size(),
             cudaMemcpyDeviceToHost);

  float ref_term = get_ref_term(type, h_weight, lambda, batch_size);
  *loss_no_val += ref_term;

  fc_layer_re.fprop(cudaStreamDefault);
  loss_re.fused_loss_computation(cudaStreamDefault);
  std::unique_ptr<float> loss_re_val(new float);
  cudaMemcpy(loss_re_val.get(), loss_tensor_re->get_ptr(), loss_tensor_re->get_size(),
             cudaMemcpyDeviceToHost);

  ASSERT_TRUE(test::compare_array_approx<float>(loss_re_val.get(), loss_no_val.get(), 1, eps));

  fc_layer_no.bprop(cudaStreamDefault);
  std::vector<float> h_wgrad_prev(wgrad_buff_no->get_num_elements());
  cudaMemcpy(&h_wgrad_prev.front(), wgrad_buff_no->get_ptr_with_offset(0),
             wgrad_buff_no->get_size(), cudaMemcpyDeviceToHost);

  cudaMemcpy(in_tensor->get_ptr(), &h_input.front(), in_tensor->get_size(), cudaMemcpyHostToDevice);
  fc_layer_re.bprop(cudaStreamDefault);
  std::vector<float> h_wgrad_next(wgrad_buff_re->get_num_elements());
  cudaMemcpy(&h_wgrad_next.front(), wgrad_buff_re->get_ptr_with_offset(0),
             wgrad_buff_re->get_size(), cudaMemcpyDeviceToHost);

  get_ref_grad(type, h_weight, h_wgrad_prev, lambda, batch_size);
  ASSERT_TRUE(test::compare_array_approx<float>(&h_wgrad_next.front(), &h_wgrad_prev.front(),
                                                h_wgrad_next.size(), eps));

  cublasDestroy(cublas_handle);
}

TEST(loss_with_regularizer, l2_32x64_64x1_small_lambda) {
  loss_with_regularizer_test(Regularizer_t::L2, 32, 64, 0.001);
}

TEST(loss_with_regularizer, l2_32x64_64x1_big_lambda) {
  loss_with_regularizer_test(Regularizer_t::L2, 32, 64, 0.1);
}

TEST(loss_with_regularizer, l2_128x256_256x1) {
  loss_with_regularizer_test(Regularizer_t::L2, 128, 256, 0.001);
}

TEST(loss_with_regularizer, l1_32x64_64x1_small_lambda) {
  loss_with_regularizer_test(Regularizer_t::L1, 32, 64, 0.001);
}

TEST(loss_with_regularizer, l1_32x64_64x1_big_lambda) {
  loss_with_regularizer_test(Regularizer_t::L1, 32, 64, 0.1);
}

TEST(loss_with_regularizer, l1_128x256_256x1) {
  loss_with_regularizer_test(Regularizer_t::L1, 128, 256, 0.001);
}

}  // namespace

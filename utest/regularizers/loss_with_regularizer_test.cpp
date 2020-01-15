/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/layers/fully_connected_layer.hpp"
#include "HugeCTR/include/loss.hpp"
#include "HugeCTR/include/regularizers/l2_regularizer.hpp"
#include "HugeCTR/include/regularizers/no_regularizer.hpp"

#include <curand.h>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <utility>

#include "cublas_v2.h"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {


const float eps = 1e-5;

void loss_with_regularizer_test(int batch_size, int num_features, float lambda) {
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  std::shared_ptr<GeneralBuffer<float>> weight_buff_no(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> weight_buff_l2(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> wgrad_buff_no(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> wgrad_buff_l2(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> blobs_buff(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> label_buff(new GeneralBuffer<float>());

  std::shared_ptr<Tensor<float>> in_tensor(
      new Tensor<float>({batch_size, num_features}, blobs_buff, TensorFormat_t::HW));

  std::shared_ptr<Tensor<float>> out_tensor(
      new Tensor<float>({batch_size, 1}, blobs_buff, TensorFormat_t::HW));

  FullyConnectedLayer fc_layer_no(weight_buff_no, wgrad_buff_no,
                               in_tensor, out_tensor,
                               TensorFormat_t::HW,
                               cublas_handle,
                               0);

  FullyConnectedLayer fc_layer_l2(weight_buff_l2, wgrad_buff_l2,
                               in_tensor, out_tensor,
                               TensorFormat_t::HW,
                               cublas_handle,
                               0);

  std::shared_ptr<Tensor<float>> loss_tensor_no(
      new Tensor<float>({1, 1}, blobs_buff, TensorFormat_t::HW));

  std::shared_ptr<Tensor<float>> loss_tensor_l2(
      new Tensor<float>({1, 1}, blobs_buff, TensorFormat_t::HW));

  std::shared_ptr<Tensor<float>> label_tensor(
      new Tensor<float>({batch_size, 1}, label_buff, TensorFormat_t::HW));

  BinaryCrossEntropyLoss loss_no(label_tensor,
                                 out_tensor,
                                 loss_tensor_no, 
                                 std::shared_ptr<NoRegularizer>(new NoRegularizer(weight_buff_no,
                                                                                  wgrad_buff_no,
                                                                                  batch_size,
                                                                                  0)),
                                 0);

  BinaryCrossEntropyLoss loss_l2(label_tensor,
                                 out_tensor,
                                 loss_tensor_l2, 
                                 std::shared_ptr<L2Regularizer>(new L2Regularizer(weight_buff_l2,
                                                                                  wgrad_buff_l2,
                                                                                  batch_size,
                                                                                  lambda,
                                                                                  cublas_handle,
                                                                                  0)),
                                 0);

  weight_buff_no->init(0);
  weight_buff_l2->init(0);
  wgrad_buff_no->init(0);
  wgrad_buff_l2->init(0);
  blobs_buff->init(0);
  label_buff->init(0);

  GaussianDataSimulator<float> input_simulator(0.0, 1.0, -1.0, 1.0);
  std::vector<float> h_input(in_tensor->get_num_elements());
  for(size_t i = 0; i < h_input.size(); i++) {
    h_input[i] = input_simulator.get_num();
  }
  cudaMemcpy(in_tensor->get_ptr(), &h_input.front(), in_tensor->get_size(),
             cudaMemcpyHostToDevice);

  float sigma = 1.f / sqrt(num_features);
  GaussianDataSimulator<float> weight_simulator(0.0, sigma, -2 * sigma, 2 * sigma);
  std::vector<float> h_weight(weight_buff_l2->get_num_elements());
  std::vector<float> h_wgrad(wgrad_buff_l2->get_num_elements());
  for(size_t i = 0; i < h_weight.size(); i++) {
    h_weight[i] = weight_simulator.get_num();
    h_wgrad[i] = weight_simulator.get_num();
  }
  cudaMemcpy(weight_buff_no->get_ptr_with_offset(0), &h_weight.front(), weight_buff_no->get_size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(weight_buff_l2->get_ptr_with_offset(0), &h_weight.front(), weight_buff_l2->get_size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(wgrad_buff_no->get_ptr_with_offset(0), &h_wgrad.front(), wgrad_buff_no->get_size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(wgrad_buff_l2->get_ptr_with_offset(0), &h_wgrad.front(), wgrad_buff_l2->get_size(),
             cudaMemcpyHostToDevice);

  UnifiedDataSimulator<int> label_simulator(0, 1);
  std::vector<float> h_label(label_tensor->get_num_elements());
  for(size_t i = 0; i < h_label.size(); i++) {
    h_label[i] = (float)label_simulator.get_num();
  }
  cudaMemcpy(label_tensor->get_ptr(), &h_label.front(), label_tensor->get_size(),
             cudaMemcpyHostToDevice);

  fc_layer_no.fprop(cudaStreamDefault);
  loss_no.fused_loss_computation(cudaStreamDefault);
  std::unique_ptr<float> loss_no_val(new float);
  cudaMemcpy(loss_no_val.get(), loss_tensor_no->get_ptr(), loss_tensor_no->get_size(),
             cudaMemcpyDeviceToHost);

  float ref_term = 0.0f;
  for(auto& v : h_weight) {
    ref_term += (v * v);
  }
  const float alpha = lambda / (batch_size * 2);
  ref_term *= alpha;

  fc_layer_l2.fprop(cudaStreamDefault);
  loss_l2.fused_loss_computation(cudaStreamDefault);
  std::unique_ptr<float> loss_l2_val(new float);
  cudaMemcpy(loss_l2_val.get(), loss_tensor_l2->get_ptr(), loss_tensor_l2->get_size(),
             cudaMemcpyDeviceToHost);

  printf("%f %f %f\n", *loss_no_val, *loss_no_val + ref_term, *loss_l2_val);

  cublasDestroy(cublas_handle);
}

TEST(loss_with_regularizer, 32x64_64x1) {
  loss_with_regularizer_test(32, 64, 0.001);
}

}

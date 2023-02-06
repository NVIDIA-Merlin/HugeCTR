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
#include <curand.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <regularizers/l2_regularizer.hpp>
#include <utest/test_utils.hpp>
#include <utility>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-5;

void l2_regularizer_test(size_t batch_size, std::vector<std::pair<size_t, size_t>> layers,
                         float lambda) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  std::shared_ptr<BufferBlock2<float>> weight_buff = buff->create_block<float>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff = buff->create_block<float>();

  Tensors2<float> weight_tensors;
  Tensors2<float> wgrad_tensors;

  for (auto& l : layers) {
    Tensor2<float> weight;
    weight_buff->reserve({l.first, l.second}, &weight);
    weight_tensors.push_back(weight);
    Tensor2<float> wgrad;
    wgrad_buff->reserve({l.first, l.second}, &wgrad);
    wgrad_tensors.push_back(wgrad);
  }

  buff->allocate();

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  std::vector<std::vector<float>> h_weights;
  for (size_t i = 0; i < layers.size(); i++) {
    auto& weight = weight_tensors[i];

    const size_t len = weight.get_num_elements();
    const size_t n_bytes = weight.get_size_in_bytes();

    std::vector<float> h_weight(len);
    simulator.fill(h_weight.data(), len);
    HCTR_LIB_THROW(cudaMemcpy(weight.get_ptr(), h_weight.data(), n_bytes, cudaMemcpyHostToDevice));
    h_weights.push_back(h_weight);
  }

  L2Regularizer<float> l2_regularizer(weight_buff->as_tensor(), wgrad_buff->as_tensor(), batch_size,
                                      lambda, test::get_default_gpu());

  // compute the regularization term
  l2_regularizer.compute_rterm();
  float out_term = l2_regularizer.get_rterm();

  float ref_term = 0.0f;
  for (const auto& h_weight : h_weights) {
    for (auto& v : h_weight) {
      ref_term += (v * v);
    }
  }
  const float alpha = lambda / (batch_size * 2);
  ref_term *= alpha;

  ASSERT_TRUE(test::compare_array_approx<float>(&out_term, &ref_term, 1, eps));

  // initialize wgard with (lambda / m) * w
  l2_regularizer.initialize_wgrad();
  for (size_t i = 0; i < layers.size(); i++) {
    const auto& wgrad = wgrad_tensors[i];
    const size_t len = wgrad.get_num_elements();
    const size_t n_bytes = wgrad.get_size_in_bytes();

    std::vector<float> out_wgrad;
    out_wgrad.resize(len);
    HCTR_LIB_THROW(
        cudaMemcpy(&out_wgrad.front(), wgrad.get_ptr(), n_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> ref_wgrad;
    for (size_t j = 0; j < len; j++) {
      ref_wgrad.push_back((lambda / batch_size) * h_weights[i][j]);
    }
    ASSERT_TRUE(test::compare_array_approx<float>(&out_wgrad.front(), &ref_wgrad.front(), 1, eps));
  }
}

TEST(l2_regularizer_layer, 32x64_64x1) { l2_regularizer_test(32, {{64, 1}}, 0.001); }

TEST(l2_regularizer_layer, 1024x64_64x256_256x1) {
  l2_regularizer_test(1024, {{64, 256}, {256, 1}}, 0.001);
}

}  // namespace

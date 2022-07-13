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

#include <curand.h>

#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/layers/fully_connected_layer.hpp"
#include "HugeCTR/include/loss.hpp"
#include "HugeCTR/include/regularizers/l1_regularizer.hpp"
#include "HugeCTR/include/regularizers/l2_regularizer.hpp"
#include "HugeCTR/include/regularizers/no_regularizer.hpp"
#include "cublas_v2.h"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

namespace {

float get_ref_term(Regularizer_t type, std::vector<float>& h_weight, float lambda,
                   size_t batch_size) {
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

std::shared_ptr<Regularizer<float>> create_regularizer(
    Regularizer_t type, const Tensor2<float>& weight_buff, const Tensor2<float>& wgrad_buff,
    size_t batch_size, float lambda, const std::shared_ptr<GPUResource>& gpu_resource) {
  std::shared_ptr<Regularizer<float>> reg;
  switch (type) {
    case Regularizer_t::L1:
      reg.reset(
          new L1Regularizer<float>(weight_buff, wgrad_buff, batch_size, lambda, gpu_resource));
      break;
    case Regularizer_t::L2:
      reg.reset(
          new L2Regularizer<float>(weight_buff, wgrad_buff, batch_size, lambda, gpu_resource));
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
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  std::shared_ptr<BufferBlock2<float>> weight_buff_no = buff->create_block<float>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff_no = buff->create_block<float>();
  std::shared_ptr<BufferBlock2<float>> weight_buff_re = buff->create_block<float>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff_re = buff->create_block<float>();

  Tensor2<float> in_tensor;
  buff->reserve({batch_size, num_features}, &in_tensor);

  Tensor2<float> out_tensor;
  buff->reserve({batch_size, 1}, &out_tensor);

  FullyConnectedLayer<float> fc_layer_no(weight_buff_no, wgrad_buff_no, in_tensor, out_tensor,
                                         test::get_default_gpu(), false, false);

  FullyConnectedLayer<float> fc_layer_re(weight_buff_re, wgrad_buff_re, in_tensor, out_tensor,
                                         test::get_default_gpu(), false, false);

  Tensor2<float> loss_tensor_no;
  buff->reserve({1, 1}, &loss_tensor_no);

  Tensor2<float> loss_tensor_re;
  buff->reserve({1, 1}, &loss_tensor_re);

  Tensor2<float> label_tensor;
  buff->reserve({batch_size, 1}, &label_tensor);

  BinaryCrossEntropyLoss<float> loss_no(
      label_tensor, out_tensor, loss_tensor_no,
      std::shared_ptr<NoRegularizer<float>>(
          new NoRegularizer<float>(weight_buff_no->as_tensor(), wgrad_buff_no->as_tensor(),
                                   batch_size, test::get_default_gpu())),
      test::get_default_gpu(), 1);
  loss_no.set_label_weight(1.0);

  BinaryCrossEntropyLoss<float> loss_re(
      label_tensor, out_tensor, loss_tensor_re,
      create_regularizer(type, weight_buff_re->as_tensor(), wgrad_buff_re->as_tensor(), batch_size,
                         lambda, test::get_default_gpu()),
      test::get_default_gpu(), 1);
  loss_re.set_label_weight(1.0);

  buff->allocate();

  test::GaussianDataSimulator input_simulator(0.0f, 1.0f);
  std::vector<float> h_input(in_tensor.get_num_elements());
  input_simulator.fill(h_input.data(), h_input.size());
  HCTR_LIB_THROW(cudaMemcpy(in_tensor.get_ptr(), &h_input.front(), in_tensor.get_size_in_bytes(),
                            cudaMemcpyHostToDevice));

  const float sigma = 1.f / sqrt(num_features);
  test::GaussianDataSimulator weight_simulator(0.0f, sigma);
  std::vector<float> h_weight(weight_buff_re->as_tensor().get_num_elements());
  weight_simulator.fill(h_weight.data(),
                        h_weight.size() % 2 != 0 ? h_weight.size() + 1 : h_weight.size());
  HCTR_LIB_THROW(cudaMemcpy(weight_buff_no->as_tensor().get_ptr(), &h_weight.front(),
                            weight_buff_no->as_tensor().get_size_in_bytes(),
                            cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(weight_buff_re->as_tensor().get_ptr(), &h_weight.front(),
                            weight_buff_re->as_tensor().get_size_in_bytes(),
                            cudaMemcpyHostToDevice));

  test::UniformDataSimulator label_simulator;
  std::vector<float> h_label(label_tensor.get_num_elements());

  label_simulator.fill(h_label.data(), h_label.size(), 0.0f, 1.0f);
  HCTR_LIB_THROW(cudaMemcpy(label_tensor.get_ptr(), &h_label.front(),
                            label_tensor.get_size_in_bytes(), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fc_layer_no.fprop(true);
  loss_no.compute_and_init(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::unique_ptr<float> loss_no_val(new float);
  HCTR_LIB_THROW(cudaMemcpy(loss_no_val.get(), loss_tensor_no.get_ptr(),
                            loss_tensor_no.get_size_in_bytes(), cudaMemcpyDeviceToHost));

  const float ref_term = get_ref_term(type, h_weight, lambda, batch_size);
  *loss_no_val += ref_term;

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fc_layer_re.fprop(true);
  loss_re.compute_and_init(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::unique_ptr<float> loss_re_val(new float);
  HCTR_LIB_THROW(cudaMemcpy(loss_re_val.get(), loss_tensor_re.get_ptr(),
                            loss_tensor_re.get_size_in_bytes(), cudaMemcpyDeviceToHost));

  ASSERT_TRUE(test::compare_array_approx<float>(loss_re_val.get(), loss_no_val.get(), 1, eps));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fc_layer_no.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::vector<float> h_wgrad_prev(wgrad_buff_no->as_tensor().get_num_elements());
  HCTR_LIB_THROW(cudaMemcpy(&h_wgrad_prev.front(), wgrad_buff_no->as_tensor().get_ptr(),
                            wgrad_buff_no->as_tensor().get_size_in_bytes(),
                            cudaMemcpyDeviceToHost));

  HCTR_LIB_THROW(cudaMemcpy(in_tensor.get_ptr(), &h_input.front(), in_tensor.get_size_in_bytes(),
                            cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fc_layer_re.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::vector<float> h_wgrad_next(wgrad_buff_re->as_tensor().get_num_elements());
  HCTR_LIB_THROW(cudaMemcpy(&h_wgrad_next.front(), wgrad_buff_re->as_tensor().get_ptr(),
                            wgrad_buff_re->as_tensor().get_size_in_bytes(),
                            cudaMemcpyDeviceToHost));

  get_ref_grad(type, h_weight, h_wgrad_prev, lambda, batch_size);
  ASSERT_TRUE(test::compare_array_approx<float>(&h_wgrad_next.front(), &h_wgrad_prev.front(),
                                                h_wgrad_next.size(), eps));
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

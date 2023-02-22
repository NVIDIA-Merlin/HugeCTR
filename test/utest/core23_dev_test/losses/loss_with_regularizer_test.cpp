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
#include <common.hpp>
#include <cstdlib>
#include <layers/fully_connected_layer.hpp>
#include <loss.hpp>
#include <regularizers/l1_regularizer.hpp>
#include <regularizers/l2_regularizer.hpp>
#include <regularizers/no_regularizer.hpp>
#include <utest/test_utils.hpp>
#include <utility>
#include <vector>

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
    Regularizer_t type, std::vector<core23::Tensor> weight_tensors,
    std::vector<core23::Tensor> wgrad_tensors, size_t batch_size, float lambda,
    const std::shared_ptr<GPUResource>& gpu_resource) {
  std::shared_ptr<Regularizer<float>> reg;
  switch (type) {
    case Regularizer_t::L1:
      reg.reset(new L1Regularizer<float>(weight_tensors, wgrad_tensors, batch_size, lambda,
                                         gpu_resource));
      break;
    case Regularizer_t::L2:
      reg.reset(new L2Regularizer<float>(weight_tensors, wgrad_tensors, batch_size, lambda,
                                         gpu_resource));
      break;
    default:
      assert(!"Error: no such optimizer && should never get here!");
      break;
  }
  return reg;
}

const float eps = 1e-5;

void loss_with_regularizer_test(Regularizer_t type, int64_t batch_size, int64_t num_features,
                                float lambda) {
  core23::BufferParams blobs_buffer_params = {};
  blobs_buffer_params.channel = GetBlobsBufferChannel();

  core23::Tensor in_tensor = core23::Tensor(core23::TensorParams()
                                                .data_type(core23::ToScalarType<float>::value)
                                                .shape({batch_size, num_features})
                                                .buffer_params(blobs_buffer_params));

  core23::Tensor out_tensor = core23::Tensor(core23::TensorParams()
                                                 .data_type(core23::ToScalarType<float>::value)
                                                 .shape({batch_size, 1})
                                                 .buffer_params(blobs_buffer_params));

  Core23TempFullyConnectedLayer<float> fc_layer_no(in_tensor, out_tensor, test::get_default_gpu(),
                                                   false, false);

  Core23TempFullyConnectedLayer<float> fc_layer_re(in_tensor, out_tensor, test::get_default_gpu(),
                                                   false, false);

  core23::Tensor loss_tensor_no = core23::Tensor(core23::TensorParams()
                                                     .data_type(core23::ToScalarType<float>::value)
                                                     .shape({1, 1})
                                                     .buffer_params(blobs_buffer_params));

  core23::Tensor loss_tensor_re = core23::Tensor(core23::TensorParams()
                                                     .data_type(core23::ToScalarType<float>::value)
                                                     .shape({1, 1})
                                                     .buffer_params(blobs_buffer_params));

  core23::Tensor label_tensor = core23::Tensor(core23::TensorParams()
                                                   .data_type(core23::ToScalarType<float>::value)
                                                   .shape({batch_size, 1})
                                                   .buffer_params(blobs_buffer_params));

  core23::Tensor empty_tensor = core23::Tensor(core23::TensorParams()
                                                   .data_type(core23::ToScalarType<float>::value)
                                                   .shape({1, 1})
                                                   .buffer_params(blobs_buffer_params));

  // may need initialize before hand
  auto weights_no = fc_layer_no.get_weights();
  auto wgrads_no = fc_layer_no.get_wgrads();

  BinaryCrossEntropyLoss<float> loss_no(
      label_tensor, out_tensor, loss_tensor_no,
      std::shared_ptr<NoRegularizer<float>>(
          new NoRegularizer<float>(weights_no, wgrads_no, batch_size, test::get_default_gpu())),
      test::get_default_gpu(), 1);
  loss_no.set_label_weight(1.0);

  auto weights_re = fc_layer_re.get_weights();
  auto wgrads_re = fc_layer_re.get_wgrads();

  BinaryCrossEntropyLoss<float> loss_re(
      label_tensor, out_tensor, loss_tensor_re,
      create_regularizer(type, weights_re, wgrads_re, batch_size, lambda, test::get_default_gpu()),
      test::get_default_gpu(), 1);
  loss_re.set_label_weight(1.0);

  core23::TensorContainer<float, 1, 1> weights_no_container(
      std::move(weights_no), {static_cast<int64_t>(weights_no.size())});

  core23::TensorContainer<float, 1, 1> wgrads_no_container(
      std::move(wgrads_no), {static_cast<int64_t>(wgrads_no.size())});

  core23::TensorContainer<float, 1, 1> weights_re_container(
      std::move(weights_re), {static_cast<int64_t>(weights_re.size())});

  core23::TensorContainer<float, 1, 1> wgrads_re_container(
      std::move(wgrads_re), {static_cast<int64_t>(wgrads_re.size())});

  auto weights_no_view = weights_no_container.flatten();
  auto weights_re_view = weights_re_container.flatten();
  auto wgrads_no_view = wgrads_no_container.flatten();
  auto wgrads_re_view = wgrads_re_container.flatten();

  test::GaussianDataSimulator input_simulator(0.0f, 1.0f);
  std::vector<float> h_input(in_tensor.num_elements());
  input_simulator.fill(h_input.data(), h_input.size());
  HCTR_LIB_THROW(cudaMemcpy(in_tensor.data(), &h_input.front(), in_tensor.num_bytes(),
                            cudaMemcpyHostToDevice));

  const float sigma = 1.f / sqrt(num_features);
  test::GaussianDataSimulator weight_simulator(0.0f, sigma);
  std::vector<float> h_weight(weights_re_view.size(0));
  weight_simulator.fill(h_weight.data(),
                        h_weight.size() % 2 != 0 ? h_weight.size() + 1 : h_weight.size());
  HCTR_LIB_THROW(cudaMemcpy(weights_no_view.data(), &h_weight.front(),
                            sizeof(float) * weights_no_view.size(0), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(weights_re_view.data(), &h_weight.front(),
                            sizeof(float) * weights_re_view.size(0), cudaMemcpyHostToDevice));

  test::UniformDataSimulator label_simulator;
  std::vector<float> h_label(label_tensor.num_elements());

  label_simulator.fill(h_label.data(), h_label.size(), 0.0f, 1.0f);
  HCTR_LIB_THROW(cudaMemcpy(label_tensor.data(), &h_label.front(), label_tensor.num_bytes(),
                            cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fc_layer_no.fprop(true);
  loss_no.compute_and_init(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::unique_ptr<float> loss_no_val(new float);
  HCTR_LIB_THROW(cudaMemcpy(loss_no_val.get(), loss_tensor_no.data(), loss_tensor_no.num_bytes(),
                            cudaMemcpyDeviceToHost));

  const float ref_term = get_ref_term(type, h_weight, lambda, batch_size);
  *loss_no_val += ref_term;

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fc_layer_re.fprop(true);
  loss_re.compute_and_init(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::unique_ptr<float> loss_re_val(new float);
  HCTR_LIB_THROW(cudaMemcpy(loss_re_val.get(), loss_tensor_re.data(), loss_tensor_re.num_bytes(),
                            cudaMemcpyDeviceToHost));

  ASSERT_TRUE(test::compare_array_approx<float>(loss_re_val.get(), loss_no_val.get(), 1, eps));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fc_layer_no.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::vector<float> h_wgrad_prev(wgrads_no_view.size(0));
  HCTR_LIB_THROW(cudaMemcpy(&h_wgrad_prev.front(), wgrads_no_view.data(),
                            sizeof(float) * wgrads_no_view.size(0), cudaMemcpyDeviceToHost));

  HCTR_LIB_THROW(cudaMemcpy(in_tensor.data(), &h_input.front(), in_tensor.num_bytes(),
                            cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fc_layer_re.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::vector<float> h_wgrad_next(wgrads_re_view.size(0));
  HCTR_LIB_THROW(cudaMemcpy(&h_wgrad_next.front(), wgrads_re_view.data(),
                            sizeof(float) * wgrads_re_view.size(0), cudaMemcpyDeviceToHost));

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

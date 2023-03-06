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

#include <gtest/gtest.h>

#include <cmath>
#include <common.hpp>
#include <core23/buffer_channel_helpers.hpp>
#include <core23/cuda_stream.hpp>
#include <core23/curand_generator.hpp>
#include <core23/data_type.hpp>
#include <core23/low_level_primitives.hpp>
#include <core23/shape.hpp>
#include <core23/tensor_container.hpp>
#include <cstdlib>
#include <regularizer_factory.hpp>
#include <utest/regularizers/regularizer_test_common.hpp>
#include <utest/test_utils.hpp>
#include <utility>

namespace HugeCTR {

namespace test {

namespace {

const float eps = 1e-5;

}

void regularizer_test_common(size_t batch_size, std::vector<core23::Shape> shapes, float lambda,
                             Regularizer_t type) {
  auto device = core23::Device::current();
  core23::CURANDGenerator generator(device);
  core23::CUDAStream stream(cudaStreamDefault, 0);

  auto weight_buffer_channel = core23::GetRandomBufferChannel();
  auto wgrad_buffer_channel = core23::GetRandomBufferChannel();

  std::vector<core23::Tensor> weight_tensor_vec;
  std::vector<core23::Tensor> wgrad_tensor_vec;
  int64_t num_elements = 0;
  for (auto shape : shapes) {
    auto tensor_params =
        core23::TensorParams().device(device).data_type(core23::ScalarType::Float).shape(shape);
    weight_tensor_vec.emplace_back(tensor_params.buffer_channel(weight_buffer_channel));
    wgrad_tensor_vec.emplace_back(tensor_params.buffer_channel(wgrad_buffer_channel));
    num_elements += shape.size();
  }

  auto regularizer = create_regularizer<float>(
      true, type, lambda, weight_tensor_vec, wgrad_tensor_vec, batch_size, test::get_default_gpu());

  int64_t num_tensors = weight_tensor_vec.size();
  WeightTensors weight_tensors(std::move(weight_tensor_vec), {num_tensors});
  WgradTensors<float> wgrad_tensors(std::move(wgrad_tensor_vec), {num_tensors});

  auto flat_weight_tensor = weight_tensors.flatten();
  auto flat_wgrad_tensor = wgrad_tensors.flatten();

  core23::normal_async<float>(flat_weight_tensor.data(), flat_weight_tensor.size(0), 0.f, 1.f,
                              device, generator, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));

  // compute the regularization term
  regularizer->compute_rterm();
  float out_term = regularizer->get_rterm();

  std::vector<float> h_weights(flat_weight_tensor.size(0));
  core23::copy_sync(h_weights.data(), flat_weight_tensor.data(),
                    flat_weight_tensor.size(0) * sizeof(float), core23::DeviceType::CPU, device);
  float ref_term = 0.f;
  if (type == Regularizer_t::L1) {
    ref_term = std::accumulate(h_weights.begin(), h_weights.end(), 0.f,
                               [](float a, float b) { return fabs(a) + fabs(b); });
    const float alpha = lambda / batch_size;
    ref_term *= alpha;
  } else if (type == Regularizer_t::L2) {
    ref_term = std::inner_product(h_weights.begin(), h_weights.end(), h_weights.begin(), 0.f);
    const float alpha = lambda / (batch_size * 2);
    ref_term *= alpha;
  } else {
  }

  ASSERT_TRUE(test::compare_array_approx<float>(&out_term, &ref_term, 1, eps));

  // initialize wgard with (lambda / m) * w
  regularizer->initialize_wgrad();
  HCTR_LIB_THROW(cudaStreamSynchronize(test::get_default_gpu()->get_stream()));
  std::vector<float> out_wgrad(flat_wgrad_tensor.size(0));
  core23::copy_sync(out_wgrad.data(), flat_wgrad_tensor.data(),
                    flat_wgrad_tensor.size(0) * sizeof(float), core23::DeviceType::CPU, device);

  std::vector<float> ref_wgrad(out_wgrad.size());
  if (type == Regularizer_t::L1) {
    std::transform(h_weights.begin(), h_weights.end(), ref_wgrad.begin(),
                   [lambda, batch_size](float w) {
                     float sign = (w > 0.f) ? 1.f : -1.f;
                     return (lambda / batch_size) * sign;
                   });
  } else if (type == Regularizer_t::L2) {
    std::transform(h_weights.begin(), h_weights.end(), ref_wgrad.begin(),
                   [lambda, batch_size](float w) { return (lambda / batch_size) * w; });
  } else {
  }
  ASSERT_TRUE(test::compare_array_approx<float>(&out_wgrad.front(), &ref_wgrad.front(),
                                                ref_wgrad.size(), eps));
}

}  // namespace test

}  // namespace HugeCTR

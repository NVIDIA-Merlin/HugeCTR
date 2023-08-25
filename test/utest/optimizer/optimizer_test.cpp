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

#include <core23/buffer_channel_helpers.hpp>
#include <core23/cuda_stream.hpp>
#include <core23/curand_generator.hpp>
#include <core23/data_type.hpp>
#include <core23/low_level_primitives.hpp>
#include <core23/shape.hpp>
#include <core23/tensor_container.hpp>
#include <general_buffer2.hpp>
#include <optimizers/adagrad_optimizer.hpp>
#include <optimizers/adam_optimizer.hpp>
#include <optimizers/ftrl_optimizer.hpp>
#include <optimizers/momentum_sgd_optimizer.hpp>
#include <optimizers/nesterov_optimizer.hpp>
#include <optimizers/sgd_optimizer.hpp>
#include <test/utest/optimizer/optimizer_cpu.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>

using namespace HugeCTR;

namespace {

template <typename T, template <typename> typename OptimizerGPU, typename... ARGS>
struct OptimizerGPUFactory {
  std::unique_ptr<Optimizer> operator()(std::vector<core23::Tensor> weight_tensors,
                                        std::vector<core23::Tensor> weight_half_tensors,
                                        std::vector<core23::Tensor> wgrad_tensors, ARGS... args) {
    WeightTensors weight_tensor_container(weight_tensors,
                                          {static_cast<int64_t>(weight_tensors.size())});
    WgradTensors<T> wgrad_tensor_container(wgrad_tensors,
                                           {static_cast<int64_t>(wgrad_tensors.size())});
    return std::make_unique<OptimizerGPU<T>>(weight_tensor_container, wgrad_tensor_container,
                                             test::get_default_gpu(), args...);
  }
};

template <typename T, typename... ARGS>
struct OptimizerGPUFactory<T, SGDOptimizer, ARGS...> {
  std::unique_ptr<Optimizer> operator()(std::vector<core23::Tensor> weight_tensors,
                                        std::vector<core23::Tensor> weight_half_tensors,
                                        std::vector<core23::Tensor> wgrad_tensors, ARGS... args) {
    WeightTensors weight_tensor_container(weight_tensors,
                                          {static_cast<int64_t>(weight_tensors.size())});
    WeightHalfTensors weight_half_tensor_container(
        weight_half_tensors, {static_cast<int64_t>(weight_half_tensors.size())});
    WgradTensors<T> wgrad_tensor_container(wgrad_tensors,
                                           {static_cast<int64_t>(wgrad_tensors.size())});
    return std::make_unique<SGDOptimizer<T>>(weight_tensor_container, weight_half_tensor_container,
                                             wgrad_tensor_container, test::get_default_gpu(),
                                             args...);
  }
};

template <typename T, template <typename> typename OptimizerGPU,
          template <typename> typename OptimizerCPU, typename... ARGS>
void optimizer_test_with_new_tensor(std::vector<core23::Shape> shapes, int num_update,
                                    float threshold, ARGS... args) {
  auto device = core23::Device::current();
  core23::CURANDGenerator generator(device);
  core23::CURANDGenerator generator_cpu(core23::DeviceType::CPU);
  core23::CUDAStream stream(cudaStreamDefault, 0);

  auto weight_buffer_channel = core23::GetRandomBufferChannel();
  auto weight_half_buffer_channel = core23::GetRandomBufferChannel();
  auto wgrad_buffer_channel = core23::GetRandomBufferChannel();

  std::vector<core23::Tensor> weight_tensor_vec;
  std::vector<core23::Tensor> weight_half_tensor_vec;
  std::vector<core23::Tensor> wgrad_tensor_vec;
  int64_t num_elements = 0;
  for (auto shape : shapes) {
    auto tensor_params = core23::TensorParams().device(device).shape(shape);
    weight_tensor_vec.emplace_back(
        tensor_params.buffer_channel(weight_buffer_channel).data_type(core23::ScalarType::Float));
    wgrad_tensor_vec.emplace_back(tensor_params.buffer_channel(wgrad_buffer_channel)
                                      .data_type(core23::ToScalarType<T>::value));
    weight_half_tensor_vec.emplace_back(tensor_params.buffer_channel(weight_half_buffer_channel)
                                            .data_type(core23::ScalarType::Half));
    num_elements += shape.size();
  }

  std::unique_ptr<Optimizer> optimizerGPU = OptimizerGPUFactory<T, OptimizerGPU, ARGS...>()(
      weight_tensor_vec, weight_half_tensor_vec, wgrad_tensor_vec, args...);

  optimizerGPU->initialize();

  int64_t num_tensors = weight_tensor_vec.size();

  WeightTensors weight_tensors(std::move(weight_tensor_vec), {num_tensors});
  WgradTensors<T> wgrad_tensors(std::move(wgrad_tensor_vec), {num_tensors});

  auto flat_weight_tensor = weight_tensors.flatten();
  auto flat_wgrad_tensor = wgrad_tensors.flatten();

  core23::normal_async<float>(flat_weight_tensor.data(), flat_weight_tensor.size(0), 0.f, 1.f,
                              device, generator, stream);

  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));

  std::unique_ptr<float[]> h_weight(new float[num_elements]);
  std::unique_ptr<T[]> h_wgrad(new T[num_elements]);
  std::unique_ptr<float[]> h_float_wgrad(new float[num_elements]);
  std::unique_ptr<float[]> h_weight_expected(new float[num_elements]);

  core23::copy_sync(h_weight_expected.get(), flat_weight_tensor.data(),
                    flat_weight_tensor.size(0) * sizeof(float), core23::DeviceType::CPU, device);

  OptimizerCPU<T> optimizerCPU(num_elements, h_weight_expected.get(), h_wgrad.get(), args...);
  for (int i = 0; i < num_update; ++i) {
    core23::normal_async<float>(h_float_wgrad.get(), num_elements, 0.f, 1.f,
                                core23::DeviceType::CPU, generator_cpu, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
    core23::convert_async<T, float>(h_wgrad.get(), h_float_wgrad.get(), num_elements,
                                    core23::DeviceType::CPU, core23::DeviceType::CPU, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
    core23::copy_sync(flat_wgrad_tensor.data(), h_wgrad.get(),
                      flat_wgrad_tensor.size(0) * sizeof(T), device, core23::DeviceType::CPU);

    HCTR_LIB_THROW(cudaDeviceSynchronize());
    optimizerGPU->update();
    optimizerCPU.update();
    HCTR_LIB_THROW(cudaDeviceSynchronize());
  }

  core23::copy_sync(h_weight.get(), flat_weight_tensor.data(),
                    flat_weight_tensor.size(0) * sizeof(float), core23::DeviceType::CPU, device);
  compare_array(h_weight.get(), h_weight_expected.get(), num_elements, threshold);
}

}  // namespace

TEST(adagrad_test, fp32_ada_grad) {
  optimizer_test_with_new_tensor<float, AdaGradOptimizer, AdaGradCPU, float, float, float, float>(
      {{64, 256}, {256, 1}}, 5, 1e-6, 1.f, 0.f, 0.f, 1);
}

TEST(adagrad_test, fp16_ada_grad) {
  optimizer_test_with_new_tensor<__half, AdaGradOptimizer, AdaGradCPU, float, float, float, float>(
      {{64, 256}, {256, 1}}, 5, 1e-3, 1.f, 0.f, 0.f, 1);
}

TEST(adam_test, fp32_adam) {
  optimizer_test_with_new_tensor<float, AdamOptimizer, AdamCPU>({{64, 256}, {256, 1}}, 5, 1e-6);
}

TEST(adam_test, fp16_adam) {
  optimizer_test_with_new_tensor<__half, AdamOptimizer, AdamCPU>({{64, 256}, {256, 1}}, 5, 1e-3);
}

TEST(ftrl_test, fp32_fltr) {
  optimizer_test_with_new_tensor<float, FtrlOptimizer, FtrlCPU>({{64, 256}, {256, 1}}, 5, 1e-6);
}

TEST(ftrl_test, fp16_fltr) {
  optimizer_test_with_new_tensor<__half, FtrlOptimizer, FtrlCPU>({{64, 256}, {256, 1}}, 5, 1e-3);
}

TEST(momentum_test, fp32_momentum) {
  optimizer_test_with_new_tensor<float, MomentumSGDOptimizer, MomentumSGDCPU, float, float, float>(
      {{64, 256}, {256, 1}}, 5, 1e-6, 0.01, 0.9, 1.f);
}

TEST(momentum_test, fp16_momentum) {
  optimizer_test_with_new_tensor<__half, MomentumSGDOptimizer, MomentumSGDCPU, float, float, float>(
      {{64, 256}, {256, 1}}, 5, 1e-3, 0.01, 0.9, 1.f);
}

TEST(nesterov, fp32_nesterov) {
  optimizer_test_with_new_tensor<float, NesterovOptimizer, NesterovCPU, float, float, float>(
      {{64, 256}, {256, 1}}, 5, 1e-6, 0.01, 0.9, 1.f);
}

TEST(nesterov, fp16_nesterov) {
  optimizer_test_with_new_tensor<__half, NesterovOptimizer, NesterovCPU, float, float, float>(
      {{64, 256}, {256, 1}}, 5, 1e-3, 0.01, 0.9, 1.f);
}

TEST(sgd, fp32_sgd) {
  optimizer_test_with_new_tensor<float, SGDOptimizer, SGDCPU>({{64, 256}, {256, 1}}, 5, 1e-6);
}

TEST(sgd, fp16_sgd) {
  optimizer_test_with_new_tensor<__half, SGDOptimizer, SGDCPU>({{64, 256}, {256, 1}}, 5, 1e-3);
}

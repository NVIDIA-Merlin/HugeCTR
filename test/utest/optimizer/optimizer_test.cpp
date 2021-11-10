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

#include "HugeCTR/include/optimizers/adagrad_optimizer.hpp"
#include "HugeCTR/include/optimizers/adam_optimizer.hpp"
#include "HugeCTR/include/optimizers/sgd_optimizer.hpp"
#include "HugeCTR/include/optimizers/momentum_sgd_optimizer.hpp"
#include "HugeCTR/include/optimizers/nesterov_optimizer.hpp"

#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/utils.hpp"
#include "gtest/gtest.h"
#include "optimizer_cpu.hpp"
#include "utest/test_utils.h"
using namespace HugeCTR;

namespace {

template <typename T, template <typename> typename OptimizerGPU, typename ... ARGS>
struct OptimizerGPUFactory {
  std::unique_ptr<Optimizer> operator()(Tensor2<float>&weight, Tensor2<__half>&weight_half, Tensor2<T>&wgrad, std::shared_ptr<BufferBlock2<T>> opt_buff, ARGS... args) {
  return std::make_unique<OptimizerGPU<T>>(
      weight, wgrad, opt_buff, test::get_default_gpu(), args...);
  }
};

template <typename T, typename ... ARGS>
struct OptimizerGPUFactory<T, SGDOptimizer, ARGS...> {
  std::unique_ptr<Optimizer> operator()(Tensor2<float>&weight, Tensor2<__half>&weight_half, Tensor2<T>&wgrad, std::shared_ptr<BufferBlock2<T>> opt_buff, ARGS... args) {
  return std::make_unique<SGDOptimizer<T>>(
      weight, weight_half, wgrad, test::get_default_gpu(), args...);
  }
};


template <typename T, template <typename> typename OptimizerGPU, template <typename> typename OptimizerCPU, typename ... ARGS>
void optimizer_test(size_t len, int num_update, float threshold, ARGS ... args) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  Tensor2<float> weight;
  buff->reserve({len}, &weight);
  Tensor2<__half> weight_half;
  buff->reserve({len}, &weight_half);
  Tensor2<T> wgrad;
  buff->reserve({len}, &wgrad);

  std::shared_ptr<BufferBlock2<T>> opt_buff = buff->create_block<T>();

  std::unique_ptr<Optimizer> optimizerGPU = OptimizerGPUFactory<T, OptimizerGPU, ARGS...>()(weight, weight_half, wgrad, opt_buff, args...);

  buff->allocate();

  optimizerGPU->initialize();

  std::unique_ptr<float[]> h_weight(new float[len]);
  std::unique_ptr<T[]> h_wgrad(new T[len]);
  std::unique_ptr<float[]> h_weight_expected(new float[len]);

  float* d_weight = weight.get_ptr();
  T* d_wgrad = wgrad.get_ptr();

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_weight.get(), len);
  for (size_t i = 0; i < len; ++i) {
    h_weight_expected[i] = h_weight[i];
  }
  cudaMemcpy(d_weight, h_weight.get(), len * sizeof(float), cudaMemcpyHostToDevice);

  OptimizerCPU<T> optimizerCPU(len, h_weight_expected.get(), h_wgrad.get(), args...);
  for (int i = 0; i < num_update; ++i) {
    simulator.fill(h_wgrad.get(), len);
    cudaMemcpy(d_wgrad, h_wgrad.get(), len * sizeof(T), cudaMemcpyHostToDevice);

    optimizerGPU->update();
    optimizerCPU.update();
  }

  cudaMemcpy(h_weight.get(), d_weight, len * sizeof(float), cudaMemcpyDeviceToHost);
  compare_array(h_weight.get(), h_weight_expected.get(), len, threshold);
}

}  // namespace

TEST(adagrad_test, fp32_ada_grad) {
  optimizer_test<float, AdaGradOptimizer, AdaGradCPU, float, float, float, float>(1024, 5, 1e-6, 1.f, 0.f, 0.f, 1);
  optimizer_test<float, AdaGradOptimizer, AdaGradCPU, float, float, float, float>(10240, 5, 1e-6, 1.f, 0.f, 0.f, 1);
}


TEST(adagrad_test, fp16_ada_grad) {
  optimizer_test<__half, AdaGradOptimizer, AdaGradCPU, float, float, float, float>(1024, 5, 1e-3, 1.f, 0.f, 0.f, 1);
  optimizer_test<__half, AdaGradOptimizer, AdaGradCPU, float, float, float, float>(10240, 5, 1e-3, 1.f, 0.f, 0.f, 1);
}

TEST(adam_test, fp32_adam) {
  optimizer_test<float, AdamOptimizer, AdamCPU>(1024, 5, 1e-6);
  optimizer_test<float, AdamOptimizer, AdamCPU>(10240, 5, 1e-6);
}


TEST(adam_test, fp16_adam) {
  optimizer_test<__half, AdamOptimizer, AdamCPU>(1024, 5, 1e-3);
  optimizer_test<__half, AdamOptimizer, AdamCPU>(10240, 5, 1e-3);
}


TEST(momentum_test, fp32_momentum) {
  optimizer_test<float, MomentumSGDOptimizer, MomentumSGDCPU, float, float, float>(1024, 5, 1e-6, 0.01, 0.9, 1.f);
  optimizer_test<float, MomentumSGDOptimizer, MomentumSGDCPU, float, float, float>(10240, 5, 1e-6, 0.01, 0.9, 1.f);
}


TEST(momentum_test, fp16_momentum) {
  optimizer_test<__half, MomentumSGDOptimizer, MomentumSGDCPU, float, float, float>(1024, 5, 1e-3, 0.01, 0.9, 1.f);
  optimizer_test<__half, MomentumSGDOptimizer, MomentumSGDCPU, float, float, float>(10240, 5, 1e-3, 0.01, 0.9,  1.f);
}

TEST(nesterov, fp32_nesterov) {
  optimizer_test<float, NesterovOptimizer, NesterovCPU, float, float, float>(1024, 5, 1e-6, 0.01, 0.9,  1.f);
  optimizer_test<float, NesterovOptimizer, NesterovCPU, float, float, float>(10240, 5, 1e-6, 0.01, 0.9, 1.f);
}


TEST(nesterov, fp16_nesterov) {
  optimizer_test<__half, NesterovOptimizer, NesterovCPU, float, float, float>(1024, 5, 1e-3, 0.01, 0.9, 1.f);
  optimizer_test<__half, NesterovOptimizer, NesterovCPU, float, float, float>(10240, 5, 1e-3, 0.01, 0.9, 1.f);
}

TEST(sgd, fp32_sgd) {
  optimizer_test<float, SGDOptimizer, SGDCPU>(1024, 5, 1e-6);
  optimizer_test<float, SGDOptimizer, SGDCPU>(10240, 5, 1e-6);
}


TEST(sgd, fp16_sgd) {
  optimizer_test<__half, SGDOptimizer, SGDCPU>(1024, 5, 1e-3);
  optimizer_test<__half, SGDOptimizer, SGDCPU>(10240, 5, 1e-3);
}


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

#include <data_simulator.hpp>
#include <diagnose.hpp>
#include <random>
#include <utils.cuh>

namespace HugeCTR {

template <>
void UniformGenerator::fill<float>(Tensor2<float>& tensor, float a, float b,
                                   const GPUResource& gpu) {
  if (a >= b) {
    CK_THROW_(Error_t::WrongInput, "a must be smaller than b");
  }

  CK_CURAND_THROW_(curandGenerateUniform(gpu.get_curand_generator(), tensor.get_ptr(),
                                         tensor.get_num_elements()));

  auto op = [a, b] __device__(float val) { return val * (b - a) + a; };
  transform_array<<<gpu.get_sm_count() * 2, 1024, 0, gpu.get_stream()>>>(
      tensor.get_ptr(), tensor.get_ptr(), tensor.get_num_elements(), op);
}

template <>
void HostUniformGenerator::fill<float>(Tensor2<float>& tensor, float a, float b,
                                       const curandGenerator_t& gen) {
  if (a >= b) {
    CK_THROW_(Error_t::WrongInput, "a must be smaller than b");
  }
  CK_CURAND_THROW_(curandGenerateUniform(gen, tensor.get_ptr(),
                                         tensor.get_num_elements() % 2 != 0
                                             ? tensor.get_num_elements() + 1
                                             : tensor.get_num_elements()));
  float* p = tensor.get_ptr();
  for (size_t i = 0; i < tensor.get_num_elements(); i++) {
    p[i] = p[i] * (b - a) + a;
  }
}

template <>
void NormalGenerator::fill<float>(Tensor2<float>& tensor, float mean, float stddev,
                                  const GPUResource& gpu) {
  CK_CURAND_THROW_(curandGenerateNormal(gpu.get_curand_generator(), tensor.get_ptr(),
                                        tensor.get_num_elements(), mean, stddev));
}

template <>
void HostNormalGenerator::fill<float>(Tensor2<float>& tensor, float mean, float stddev,
                                      const curandGenerator_t& gen) {
  CK_CURAND_THROW_(curandGenerateNormal(gen, tensor.get_ptr(),
                                        tensor.get_num_elements() % 2 != 0
                                            ? tensor.get_num_elements() + 1
                                            : tensor.get_num_elements(),
                                        mean, stddev));
}

void ConstantDataSimulator::fill(Tensor2<float>& tensor, const curandGenerator_t& gen) {
  float* p = tensor.get_ptr();
  for (size_t i = 0; i < tensor.get_num_elements(); i++) {
    p[i] = value_;
  }
}

void UniformDataSimulator::fill(Tensor2<float>& tensor, const curandGenerator_t& gen) {
  HostUniformGenerator::fill(tensor, min_, max_, gen);
}

void GaussianDataSimulator::fill(Tensor2<float>& tensor, const curandGenerator_t& gen) {
  HostNormalGenerator::fill(tensor, mu_, sigma_, gen);
}
}  // namespace HugeCTR
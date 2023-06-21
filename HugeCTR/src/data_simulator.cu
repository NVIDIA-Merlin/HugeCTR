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

#include <data_simulator.hpp>
#include <diagnose.hpp>
#include <random>
#include <utils.cuh>

namespace HugeCTR {

template <>
void UniformGenerator::fill<float>(float* ptr, size_t num_elements, float a, float b,
                                   size_t sm_count, const curandGenerator_t& generator,
                                   const cudaStream_t& stream) {
  if (a >= b) {
    HCTR_OWN_THROW(Error_t::WrongInput, "a must be smaller than b");
  }

  HCTR_LIB_THROW(curandGenerateUniform(generator, ptr, num_elements));

  auto op = [a, b] __device__(float val) { return val * (b - a) + a; };
  transform_array<<<sm_count * 2, 1024, 0, stream>>>(ptr, ptr, num_elements, op);
}

template <typename T>
__global__ void sinusoidal_kernel(T* output, int ev_size, int max_sequence_len) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  int offset = row * ev_size + col;
  float log_result = __logf(10000) / (ev_size);
  float exp_result = __expf(((col >> 1) << 1) * -1 * log_result);

  if (col < ev_size) {
    output[offset] = (col % 2) ? (T)__cosf(exp_result * row) : (T)__sinf(exp_result * row);
  }
}

template <>
void SinusoidalGenerator::fill<float>(float* ptr, size_t num_elements, int ev_size,
                                      int max_sequence_len, size_t sm_count,
                                      const cudaStream_t& stream) {
  sinusoidal_kernel<<<max_sequence_len, max(32, ev_size), 0, stream>>>(ptr, ev_size,
                                                                       max_sequence_len);
}

// TODO remove Tensor2 fill
template <>
void UniformGenerator::fill<float>(Tensor2<float>& tensor, float a, float b, size_t sm_count,
                                   const curandGenerator_t& generator, const cudaStream_t& stream) {
  UniformGenerator::fill<float>(tensor.get_ptr(), tensor.get_num_elements(), a, b, sm_count,
                                generator, stream);
}

// TODO remove Tensor2 fill
template <>
void HostUniformGenerator::fill<float>(Tensor2<float>& tensor, float a, float b,

                                       const curandGenerator_t& generator) {
  if (a >= b) {
    HCTR_OWN_THROW(Error_t::WrongInput, "a must be smaller than b");
  }
  HCTR_LIB_THROW(curandGenerateUniform(generator, tensor.get_ptr(),
                                       tensor.get_num_elements() % 2 != 0
                                           ? tensor.get_num_elements() + 1
                                           : tensor.get_num_elements()));
  float* p = tensor.get_ptr();
  for (size_t i = 0; i < tensor.get_num_elements(); i++) {
    p[i] = p[i] * (b - a) + a;
  }
}
// TODO remove Tensor2 fill
template <>
void NormalGenerator::fill<float>(Tensor2<float>& tensor, float mean, float stddev, size_t sm_count,
                                  const curandGenerator_t& generator, const cudaStream_t& stream) {
  HCTR_LIB_THROW(
      curandGenerateNormal(generator, tensor.get_ptr(), tensor.get_num_elements(), mean, stddev));
}

// TODO remove Tensor2 fill
template <>
void HostNormalGenerator::fill<float>(Tensor2<float>& tensor, float mean, float stddev,
                                      const curandGenerator_t& gen) {
  HCTR_LIB_THROW(curandGenerateNormal(gen, tensor.get_ptr(),
                                      tensor.get_num_elements() % 2 != 0
                                          ? tensor.get_num_elements() + 1
                                          : tensor.get_num_elements(),
                                      mean, stddev));
}

// TODO remove Tensor2 fill
void ConstantDataSimulator::fill(Tensor2<float>& tensor, const curandGenerator_t& gen) {
  float* p = tensor.get_ptr();
  for (size_t i = 0; i < tensor.get_num_elements(); i++) {
    p[i] = value_;
  }
}

// TODO remove Tensor2 fill
void UniformDataSimulator::fill(Tensor2<float>& tensor, const curandGenerator_t& gen) {
  HostUniformGenerator::fill(tensor, min_, max_, gen);
}

// TODO remove Tensor2 fill
void GaussianDataSimulator::fill(Tensor2<float>& tensor, const curandGenerator_t& gen) {
  HostNormalGenerator::fill(tensor, mu_, sigma_, gen);
}

template <>
void UniformGenerator::fill<float>(core23::Tensor& tensor, float a, float b, size_t sm_count,
                                   const curandGenerator_t& generator, const cudaStream_t& stream) {
  UniformGenerator::fill<float>(tensor.data<float>(), tensor.num_elements(), a, b, sm_count,
                                generator, stream);
}

template <>
void HostUniformGenerator::fill<float>(core23::Tensor& tensor, float a, float b,
                                       const curandGenerator_t& generator) {
  if (a >= b) {
    HCTR_OWN_THROW(Error_t::WrongInput, "a must be smaller than b");
  }
  int64_t num_elements = tensor.num_elements();
  int64_t even_length = tensor.num_elements() / 2 * 2;
  float* p = tensor.data<float>();
  float tmp[2];
  // in case the length is odd
  if (num_elements & 1) {
    float* last_element_ptr = tensor.data<float>() + even_length;
    HCTR_LIB_THROW(curandGenerateUniform(generator, tmp, 2));
    tmp[0] = tmp[0] * (b - a) + a;
    *last_element_ptr = tmp[0];
  }
  // even_length=0 is allowed
  HCTR_LIB_THROW(curandGenerateUniform(generator, tensor.data<float>(), even_length));
  for (int64_t i = 0; i < tensor.num_elements(); i++) {
    p[i] = p[i] * (b - a) + a;
  }
}

template <>
void NormalGenerator::fill<float>(core23::Tensor& tensor, float mean, float stddev, size_t sm_count,
                                  const curandGenerator_t& generator, const cudaStream_t& stream) {
  HCTR_LIB_THROW(
      curandGenerateNormal(generator, tensor.data<float>(), tensor.num_elements(), mean, stddev));
}

template <>
void HostNormalGenerator::fill<float>(core23::Tensor& tensor, float mean, float stddev,
                                      const curandGenerator_t& gen) {
  int64_t num_elements = tensor.num_elements();
  int64_t even_length = tensor.num_elements() / 2 * 2;
  float* p = tensor.data<float>();
  float tmp[2];
  if (num_elements & 1) {
    float* last_element_ptr = p + even_length;
    HCTR_LIB_THROW(curandGenerateNormal(gen, tmp, 2, mean, stddev));
    *last_element_ptr = tmp[0];
  }

  HCTR_LIB_THROW(curandGenerateNormal(gen, p, even_length, mean, stddev));
}

void ConstantDataSimulator::fill(core23::Tensor& tensor, const curandGenerator_t& gen) {
  float* p = tensor.data<float>();
  for (int64_t i = 0; i < tensor.num_elements(); i++) {
    p[i] = value_;
  }
}

void UniformDataSimulator::fill(core23::Tensor& tensor, const curandGenerator_t& gen) {
  HostUniformGenerator::fill<float>(tensor, min_, max_, gen);
}

void GaussianDataSimulator::fill(core23::Tensor& tensor, const curandGenerator_t& gen) {
  HostNormalGenerator::fill<float>(tensor, mu_, sigma_, gen);
}

}  // namespace HugeCTR

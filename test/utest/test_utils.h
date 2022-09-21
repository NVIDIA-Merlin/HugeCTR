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

#pragma once

#include <common.hpp>
#include <random>

#include "HugeCTR/include/gpu_resource.hpp"
#include "gtest/gtest.h"

namespace HugeCTR {

namespace test {

template <typename T>
T abs(const T& val) {
  return val > T(0) ? val : -val;
}

template <typename T>
::testing::AssertionResult compare_array_approx(const T* h_out, const T* h_exp, int len, T eps) {
  for (int i = 0; i < len; ++i) {
    auto output = h_out[i];
    auto expected = h_exp[i];
    T diff = abs(output - expected);
    if (diff > eps) {
      // if (diff > eps && i < 128 * 10) {
      // std::cout
      return ::testing::AssertionFailure() << "output: " << output << " != expected: " << expected
                                           << " at idx " << i << std::endl;
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename T>
::testing::AssertionResult compare_array_approx(const T* h_out, const T expected, int len, T eps) {
  for (int i = 0; i < len; ++i) {
    auto output = h_out[i];
    T diff = abs(output - expected);
    if (diff > eps) {
      return ::testing::AssertionFailure()
             << "output: " << output << " != expected: " << expected << " at idx " << i;
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename T>
::testing::AssertionResult compare_array_approx_with_ratio(const T* h_out, const T* h_exp, int len,
                                                           T eps, float ratio) {
  int mismatch_tolerable_len = len * ratio;
  for (int i = 0; i < len && mismatch_tolerable_len > 0; ++i) {
    auto output = h_out[i];
    auto expected = h_exp[i];
    T diff = abs(output - expected);
    if (diff > eps) mismatch_tolerable_len--;
  }
  if (mismatch_tolerable_len == 0) {
    return ::testing::AssertionFailure() << "Mismatch ratio is larger than tolerance";
  } else {
    return ::testing::AssertionSuccess();
  }
}

template <typename T>
::testing::AssertionResult compare_array_approx_rel(const T* h_out, const T* h_exp, int len,
                                                    T max_rel_err, T max_abs_err) {
  for (int i = 0; i < len; ++i) {
    auto output = h_out[i];
    auto expected = h_exp[i];
    T abs_err = abs(output - expected);
    T rel_err = abs_err / expected;
    if (abs_err > max_abs_err && rel_err > max_rel_err) {
      return ::testing::AssertionFailure()
             << "output: " << output << " != expected: " << expected << " at idx " << i;
    }
  }
  return ::testing::AssertionSuccess();
}

__forceinline__ bool cpu_gpu_cmp(float* cpu_p, float* gpu_p, int len) {
  float* gpu_tmp = (float*)malloc(sizeof(float) * len);
  cudaMemcpy(gpu_tmp, gpu_p, sizeof(float) * len, cudaMemcpyDeviceToHost);
  bool flag = true;
  for (int i = 0; i < len; ++i) {
    if (fabs(gpu_tmp[i] - cpu_p[i]) >= 1e-5) {
      HCTR_LOG(INFO, WORLD, "gpu_tmp(%f) - cpu_p(%f) >= 1e-5 when i = %d\n", gpu_tmp[i], cpu_p[i],
               i);
      flag = false;
      break;
    }
  }
  free(gpu_tmp);
  return flag;
}

__forceinline__ void mpi_init() {
#ifdef ENABLE_MPI
  int flag = 0;
  MPI_Initialized(&flag);
  if (!flag) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }
#endif
}

__forceinline__ void mpi_finalize() {
#ifdef ENABLE_MPI
  MPI_Finalize();
#endif
}

inline size_t align_to_even(size_t n) { return (n % 2 != 0) ? n + 1 : n; }

class GaussianDataSimulator {
  float mean_, stddev_;
  curandGenerator_t curand_generator_;

 public:
  GaussianDataSimulator(float mean, float stddev) : mean_(mean), stddev_(stddev) {
    HCTR_LIB_THROW(curandCreateGeneratorHost(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  }

  ~GaussianDataSimulator() { curandDestroyGenerator(curand_generator_); }

  void fill(float* arr, size_t len) {
    HCTR_LIB_THROW(curandGenerateNormal(curand_generator_, arr, len, mean_, stddev_));
  }

  void fill(__half* arr, size_t len) {
    std::unique_ptr<float[]> farr(new float[len]);
    HCTR_LIB_THROW(curandGenerateNormal(curand_generator_, farr.get(), len, mean_, stddev_));
    for (size_t i = 0; i < len; i++) {
      arr[i] = __float2half(farr[i]);
    }
  }
};

class UniformDataSimulator {
  curandGenerator_t curand_generator_;

 public:
  UniformDataSimulator() {
    HCTR_LIB_THROW(curandCreateGeneratorHost(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  }

  ~UniformDataSimulator() { curandDestroyGenerator(curand_generator_); }

  void fill(int* arr, size_t len, int a, int b) {
    if (a >= b) {
      HCTR_OWN_THROW(Error_t::WrongInput, "a must be smaller than b");
    }
    HCTR_LIB_THROW(curandGenerate(curand_generator_, reinterpret_cast<unsigned int*>(arr), len));
    for (size_t i = 0; i < len; i++) {
      arr[i] = arr[i] % (b - a) + a;
    }
  }

  void fill(float* arr, size_t len, float a, float b) {
    if (a >= b) {
      HCTR_OWN_THROW(Error_t::WrongInput, "a must be smaller than b");
    }
    HCTR_LIB_THROW(curandGenerateUniform(curand_generator_, arr, len));
    for (size_t i = 0; i < len; i++) {
      arr[i] = arr[i] * (b - a) + a;
    }
  }
};

static std::shared_ptr<GPUResource> get_default_gpu() {
  std::random_device rd;
  return std::make_shared<GPUResource>(0, 0, 0, rd(), rd(), nullptr);
}

}  // end namespace test

}  // namespace HugeCTR

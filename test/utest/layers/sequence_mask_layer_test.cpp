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

#include "HugeCTR/include/layers/sequence_mask_layer.hpp"

#include <gtest/gtest.h>
#include <utest/test_utils.h>

#include <vector>

#include "HugeCTR/include/utils.hpp"

using namespace HugeCTR;

namespace {

template <typename T>
struct Eps {
  static T value();
};

template <>
struct Eps<float> {
  static constexpr float value() { return 1e-6f; }
};

template <>
struct Eps<__half> {
  static __half value() { return __float2half(1e-2f); }
};

template <typename T>
void f2i_input(T* input, size_t in_size, size_t max_sequence_len) {
  for (size_t i = 0; i < in_size; i++) {
    input[i] = abs(floor(input[i] * max_sequence_len));
  }
}

template <typename T>
void sequence_mask_cpu(T* input, T* output, size_t batch_size, size_t max_sequence_len,
                       size_t out_size) {
  for (size_t i = 0; i < batch_size; i++) {
    float length = input[i];
    for (size_t j = 0; j < max_sequence_len; j++) {
      if (j < length) {
        output[i * max_sequence_len + j] = (T)(1.0f);
      } else {
        output[i * max_sequence_len + j] = (T)(0.0f);
      }
    }
  }
}

template <typename T>
void sequence_mask_test(size_t batch_size, size_t max_sequence_len) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  std::vector<size_t> dim_in = {batch_size};
  std::vector<size_t> dim_out = {batch_size, 1, 1, max_sequence_len};
  size_t in_size = batch_size;
  size_t out_size = batch_size * max_sequence_len;

  Tensor2<T> in_tensor;
  buff->reserve(dim_in, &in_tensor);
  Tensor2<T> out_tensor;
  buff->reserve(dim_out, &out_tensor);

  SequenceMaskLayer<T> sequence_mask_layer(in_tensor, out_tensor, max_sequence_len, buff,
                                           test::get_default_gpu());

  buff->allocate();

  T* h_d_in = in_tensor.get_ptr();

  T* d_out = out_tensor.get_ptr();

  std::unique_ptr<T[]> h_in(new T[in_size]);
  std::unique_ptr<T[]> h_d_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop

  simulator.fill(h_in.get(), in_size);
  f2i_input(h_in.get(), in_size, max_sequence_len);
  HCTR_LIB_THROW(cudaMemcpy(h_d_in, h_in.get(), in_size * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  sequence_mask_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(cudaMemcpy(h_d_out.get(), d_out, out_size * sizeof(T), cudaMemcpyDeviceToHost));

  sequence_mask_cpu(h_in.get(), h_cpu_out.get(), batch_size, max_sequence_len, out_size);
  ASSERT_TRUE(
      test::compare_array_approx<T>(h_d_out.get(), h_cpu_out.get(), out_size, Eps<T>::value()));
}

}  // namespace

TEST(sequence_mask_layer, fp32_8192x200) { sequence_mask_test<float>(8192, 200); }
TEST(sequence_mask_layer, fp16_8192x1000) { sequence_mask_test<__half>(8192, 1000); }
TEST(sequence_mask_layer, fp32_8192x800) { sequence_mask_test<float>(4, 800); }
TEST(sequence_mask_layer, fp16_8192x40) { sequence_mask_test<__half>(8192, 40); }
TEST(sequence_mask_layer, fp32_4096x40) { sequence_mask_test<float>(4096, 40); }
TEST(sequence_mask_layer, fp16_4096x400) { sequence_mask_test<__half>(4096, 400); }

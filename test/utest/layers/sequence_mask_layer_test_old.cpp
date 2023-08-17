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

#include <layers/sequence_mask_layer.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

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
    input[i] = abs(floor(__half2float(input[i]) * max_sequence_len));
  }
}

template <typename T>
void sequence_mask_cpu(T* seq_len_from, T* seq_len_to, T* output, size_t batch_size,
                       size_t max_sequence_len_from, size_t max_sequence_len_to, size_t out_size) {
  for (size_t i = 0; i < batch_size; i++) {
    float length_from = seq_len_from[i];
    float length_to = seq_len_to[i];
    for (size_t j = 0; j < max_sequence_len_from; j++) {
      for (size_t k = 0; k < max_sequence_len_to; k++) {
        if (j < length_from && k < length_to) {
          output[i * max_sequence_len_from * max_sequence_len_to + j * max_sequence_len_to + k] =
              (T)(1.0f);
        } else {
          output[i * max_sequence_len_from * max_sequence_len_to + j * max_sequence_len_to + k] =
              (T)(0.0f);
        }
      }
    }
  }
}

template <typename T>
void sequence_mask_test(size_t batch_size, size_t max_sequence_len_from,
                        size_t max_sequence_len_to) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  std::vector<size_t> dim_in = {batch_size};
  std::vector<size_t> dim_out = {batch_size, 1, max_sequence_len_from, max_sequence_len_to};
  size_t in_size = batch_size;
  size_t out_size = batch_size * max_sequence_len_from * max_sequence_len_to;

  Tensors2<T> in_tensors;
  Tensor2<T> seq_from_tensor;
  buff->reserve(dim_in, &seq_from_tensor);
  in_tensors.push_back(seq_from_tensor);
  Tensor2<T> seq_to_tensor;
  buff->reserve(dim_in, &seq_to_tensor);
  in_tensors.push_back(seq_to_tensor);

  Tensor2<T> out_tensor;
  buff->reserve(dim_out, &out_tensor);

  SequenceMaskLayer<T> sequence_mask_layer(in_tensors, out_tensor, max_sequence_len_from,
                                           max_sequence_len_to, buff, test::get_default_gpu());

  buff->allocate();

  T* h_d_from_in = in_tensors[0].get_ptr();
  T* h_d_to_in = in_tensors[1].get_ptr();

  T* d_out = out_tensor.get_ptr();

  std::unique_ptr<T[]> h_from_in(new T[in_size]);
  std::unique_ptr<T[]> h_to_in(new T[in_size]);
  std::unique_ptr<T[]> h_d_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop

  simulator.fill(h_from_in.get(), in_size);
  simulator.fill(h_to_in.get(), in_size);
  f2i_input(h_from_in.get(), in_size, max_sequence_len_from);
  f2i_input(h_to_in.get(), in_size, max_sequence_len_to);
  HCTR_LIB_THROW(
      cudaMemcpy(h_d_from_in, h_from_in.get(), in_size * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(h_d_to_in, h_to_in.get(), in_size * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  sequence_mask_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(cudaMemcpy(h_d_out.get(), d_out, out_size * sizeof(T), cudaMemcpyDeviceToHost));

  sequence_mask_cpu(h_from_in.get(), h_to_in.get(), h_cpu_out.get(), batch_size,
                    max_sequence_len_from, max_sequence_len_to, out_size);
  ASSERT_TRUE(
      test::compare_array_approx<T>(h_d_out.get(), h_cpu_out.get(), out_size, Eps<T>::value()));
}

}  // namespace

TEST(sequence_mask_layer_old, fp32_8192x200) { sequence_mask_test<float>(2, 20, 20); }
TEST(sequence_mask_layer_old, fp16_8192x1000) { sequence_mask_test<__half>(8192, 1000, 800); }
TEST(sequence_mask_layer_old, fp32_8192x800) { sequence_mask_test<float>(4, 800, 800); }
TEST(sequence_mask_layer_old, fp16_8192x40) { sequence_mask_test<__half>(8192, 40, 100); }
TEST(sequence_mask_layer_old, fp32_4096x40) { sequence_mask_test<float>(4096, 40, 20); }
TEST(sequence_mask_layer_old, fp16_4096x400) { sequence_mask_test<__half>(4096, 400, 400); }

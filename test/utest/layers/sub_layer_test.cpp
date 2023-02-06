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

#include <layers/sub_layer.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-6f;

template <typename T>
void sub_cpu(T **input, T *output, size_t size) {
  for (size_t i = 0; i < size; i++) output[i] = input[0][i] - input[1][i];
}

template <typename T>
void sub_dgrad_cpu(const T *top_grad, T **dgrad, size_t size) {
  for (size_t i = 0; i < size; i++) {
    dgrad[0][i] = top_grad[i];
    dgrad[1][i] = 0.0 - top_grad[i];
  }
}

template <typename T>
void sub_test(size_t batch_size, size_t slot_num, size_t embedding_vec_size, size_t num) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  std::vector<size_t> dims_in = {batch_size, slot_num, embedding_vec_size};
  std::vector<size_t> dims_out = {batch_size, slot_num, embedding_vec_size};
  size_t size = batch_size * slot_num * embedding_vec_size;

  Tensors2<T> in_tensors;
  for (size_t i = 0; i < num; i++) {
    Tensor2<T> in_tensor;
    buff->reserve(dims_in, &in_tensor);
    in_tensors.push_back(in_tensor);
  }
  Tensor2<T> out_tensor;
  buff->reserve(dims_out, &out_tensor);

  SubLayer<T> sub_layer(in_tensors, out_tensor, buff, test::get_default_gpu());

  buff->allocate();
  sub_layer.initialize();

  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }
  T *d_out = out_tensor.get_ptr();

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_ins[i] = new T[size];
  }
  std::unique_ptr<T[]> h_out(new T[size]);
  std::unique_ptr<T[]> h_cpu_out(new T[size]);
  std::unique_ptr<T *[]> h_gpu_dgrads(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_gpu_dgrads[i] = new T[size];
  }

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop
  for (size_t i = 0; i < num; i++) {
    simulator.fill(h_ins[i], size);
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  sub_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(cudaMemcpy(h_out.get(), d_out, size * sizeof(T), cudaMemcpyDeviceToHost));

  sub_cpu(h_ins.get(), h_cpu_out.get(), size);
  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), size, eps));

  // bprop
  for (size_t i = 0; i < num; i++) {
    simulator.fill(h_ins[i], size);
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }
  simulator.fill(h_out.get(), size);
  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), size * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  sub_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  for (size_t i = 0; i < num; i++) {
    HCTR_LIB_THROW(
        cudaMemcpy(h_gpu_dgrads[i], h_d_ins[i], size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  sub_dgrad_cpu(h_out.get(), h_ins.get(), size);
  for (size_t i = 0; i < num; i++) {
    ASSERT_TRUE(
        test::compare_array_approx<T>(h_ins[i], h_gpu_dgrads[i], size, eps));  // compare dgrad
  }
}

}  // namespace

TEST(sub_layer, fp32) { sub_test<float>(40960, 1, 1, 2); }
TEST(sub_layer, fp16) { sub_test<float>(40960, 2, 110, 2); }

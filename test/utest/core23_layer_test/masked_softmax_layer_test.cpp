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

#include <layers/masked_softmax_layer.hpp>
#include <layers/sequence_mask_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;
using namespace std;

namespace {

const float eps = 1e-4;

template <typename T>
void f2i_input(T* input, size_t in_size, size_t max_sequence_len) {
  for (size_t i = 0; i < in_size; i++) {
    input[i] = abs(floor(input[i] * max_sequence_len));
  }
}

template <typename T>
void max_per_line_cpu(T* bottom, const T* mask, int batch_size, int head_num, int seq_len_from,
                      int seq_len_to, float scalar, T* workspace) {
  float local_max = -1e20f;
  for (int i = 0; i < batch_size * head_num * seq_len_from; i++) {
    local_max = -1e20f;
    int input_offset = i * seq_len_to;
    int mask_offset = (i / (head_num * seq_len_from)) * seq_len_from * seq_len_to +
                      (i % seq_len_from) * seq_len_to;
    for (int j = 0; j < seq_len_to; j++) {
      float in_val = static_cast<float>(bottom[input_offset + j]);
      float mask_val = (float)mask[mask_offset + j];
      mask_val = (1.0f - mask_val) * 10000.0f;
      bottom[input_offset + j] = in_val * scalar - (float)mask_val;
      local_max = max(local_max, bottom[input_offset + j]);
    }
    workspace[i] = static_cast<T>(local_max);
  }
}

template <typename T>
void sum_ex_cpu(T* top, int embedding_vector_size, int dim0, T* workspace) {
  // sum(e^xi) i = [0, embedding_vector_size -1];
  for (int i = 0; i < dim0; i++) {
    workspace[i] = 0;
    int offset = i * embedding_vector_size;
    for (int j = 0; j < embedding_vector_size; j++) {
      workspace[i] += top[offset + j];
    }
  }
}

template <typename T>
void ex_cpu(T* top, const T* bottom, const T* workspace, int dim0, int embedding_vector_size) {
  // e^xi
  for (int i = 0; i < dim0; i++) {
    int offset = i * embedding_vector_size;
    for (int j = 0; j < embedding_vector_size; j++) {
      top[offset + j] = expf(bottom[offset + j] - workspace[i]);
    }
  }
}

template <typename T>
void sum_grad_softmax(const T* d_top, const T* softmax_out, int embedding_vector_size, int dim0,
                      T* workspace) {
  for (int i = 0; i < dim0; i++) {
    float grad_sum = 0.0;
    int offset = i * embedding_vector_size;
    for (int j = 0; j < embedding_vector_size; j++) {
      grad_sum += (float)(d_top[offset + j] * softmax_out[offset + j]);
    }
    workspace[i] = static_cast<T>(grad_sum);
  }
}

template <typename T>
void masked_softmax_fprop_cpu(T* top, T* bottom, const T* mask, int batch_size, int head_num,
                              int seq_len_from, int seq_len_to, float scalar) {
  int dim0 = batch_size * head_num * seq_len_from;
  T* workspace = new T[batch_size * head_num * seq_len_from];

  // max per line
  max_per_line_cpu(bottom, mask, batch_size, head_num, seq_len_from, seq_len_to, scalar, workspace);

  // e^xi
  ex_cpu(top, bottom, workspace, dim0, seq_len_to);
  // sum(e^xi) i = [0, embedding_vector_size -1];
  sum_ex_cpu(top, seq_len_to, dim0, workspace);

  // softmax : e^xi / sum(e^xi); i = [0, len - 1];
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < seq_len_to; j++) {
      int index = i * seq_len_to + j;
      top[index] = top[index] / workspace[i];
    }
  }
  delete[] workspace;
}

template <typename T>
void masked_softmax_bprop_cpu(T* d_bottom, const T* d_top, const T* softmax_out, int dim0,
                              int embedding_vector_size, float scalar) {
  T* workspace = new T[dim0];

  sum_grad_softmax(d_top, softmax_out, embedding_vector_size, dim0, workspace);
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < embedding_vector_size; j++) {
      int index = i * embedding_vector_size + j;
      d_bottom[index] = softmax_out[index] * (d_top[index] - workspace[i]);
      d_bottom[index] = d_bottom[index] * scalar;
    }
  }
  delete[] workspace;
}
template <typename T>
void core23_masked_softmax_test(int64_t batch_size, int64_t head_num, int64_t seq_len_from,
                                int64_t seq_len_to, float scalar) {
  core23::Shape dims_output = {batch_size, head_num, seq_len_from, seq_len_to};
  core23::Shape dims_input = {batch_size, head_num, seq_len_from, seq_len_to};
  core23::Shape dims_mask = {batch_size, 1, seq_len_from, seq_len_to};
  core23::Shape dims_input_len = {batch_size};

  core23::Tensor input_len_from_tensor(core23::TensorParams()
                                           .shape(dims_input_len)
                                           .data_type(core23::ToScalarType<T>::value)
                                           .device({core23::DeviceType::GPU, 0}));

  core23::Tensor input_len_to_tensor(core23::TensorParams()
                                         .shape(dims_input_len)
                                         .data_type(core23::ToScalarType<T>::value)
                                         .device({core23::DeviceType::GPU, 0}));

  std::vector<core23::Tensor> mask_input_tensors;
  mask_input_tensors.push_back(input_len_from_tensor);
  mask_input_tensors.push_back(input_len_to_tensor);

  std::vector<core23::Tensor> bottom_tensors;
  core23::Tensor mask_tensor(core23::TensorParams()
                                 .shape(dims_mask)
                                 .data_type(core23::ToScalarType<T>::value)
                                 .device({core23::DeviceType::GPU, 0}));
  core23::Tensor input_tensor(core23::TensorParams()
                                  .shape(dims_input)
                                  .data_type(core23::ToScalarType<T>::value)
                                  .device({core23::DeviceType::GPU, 0}));

  bottom_tensors.push_back(input_tensor);
  bottom_tensors.push_back(mask_tensor);

  core23::Tensor top_tensor(core23::TensorParams()
                                .shape(dims_output)
                                .data_type(core23::ToScalarType<T>::value)
                                .device({core23::DeviceType::GPU, 0}));

  MaskedSoftmaxLayer<T> masked_softmax_layer(bottom_tensors, top_tensor, scalar,
                                             test::get_default_gpu());
  SequenceMaskLayer<T> sequence_mask_layer(mask_input_tensors, mask_tensor, seq_len_from,
                                           seq_len_to, test::get_default_gpu());

  const int64_t tensor_size = batch_size * head_num * seq_len_from * seq_len_to;

  std::unique_ptr<T[]> h_in_from_len(new T[batch_size]);
  std::unique_ptr<T[]> h_in_to_len(new T[batch_size]);
  std::unique_ptr<T[]> h_mask(new T[batch_size * seq_len_from * seq_len_to]);
  std::unique_ptr<T[]> h_bottom(new T[tensor_size]);
  std::unique_ptr<T[]> h_top(new T[tensor_size]);
  std::unique_ptr<T[]> h_softmax_out(new T[tensor_size]);
  std::unique_ptr<T[]> d2h_top(new T[tensor_size]);
  std::unique_ptr<T[]> h_bottom_grad(new T[tensor_size]);
  std::unique_ptr<T[]> d2h_bottom_grad(new T[tensor_size]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_in_from_len.get(), batch_size);
  f2i_input(h_in_from_len.get(), batch_size, seq_len_from);
  simulator.fill(h_in_to_len.get(), batch_size);
  f2i_input(h_in_to_len.get(), batch_size, seq_len_to);

  simulator.fill(h_bottom.get(), tensor_size);

  // generate sequence mask
  HCTR_LIB_THROW(cudaMemcpy(input_len_from_tensor.data(), h_in_from_len.get(),
                            batch_size * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(input_len_to_tensor.data(), h_in_to_len.get(), batch_size * sizeof(T),
                            cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(input_tensor.data(), h_bottom.get(), tensor_size * sizeof(T),
                            cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  sequence_mask_layer.fprop(true);
  HCTR_LIB_THROW(cudaMemcpy(h_mask.get(), mask_tensor.data(),
                            batch_size * seq_len_from * seq_len_to * sizeof(T),
                            cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  masked_softmax_layer.fprop(true);

  HCTR_LIB_THROW(cudaMemcpy(d2h_top.get(), top_tensor.data(), tensor_size * sizeof(T),
                            cudaMemcpyDeviceToHost));

  masked_softmax_fprop_cpu<T>(h_top.get(), h_bottom.get(), h_mask.get(), batch_size, head_num,
                              seq_len_from, seq_len_to, scalar);

  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.get(), h_top.get(), tensor_size, eps));

  // bprop
  simulator.fill(h_top.get(), tensor_size);
  masked_softmax_fprop_cpu<T>(h_softmax_out.get(), h_bottom.get(), h_mask.get(), batch_size,
                              head_num, seq_len_from, seq_len_to, scalar);

  HCTR_LIB_THROW(
      cudaMemcpy(top_tensor.data(), h_top.get(), tensor_size * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(masked_softmax_layer.get_softmax_tensor().data(), h_softmax_out.get(),
                            tensor_size * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  masked_softmax_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaMemcpy(d2h_bottom_grad.get(), input_tensor.data(), tensor_size * sizeof(T),
                            cudaMemcpyDeviceToHost));
  masked_softmax_bprop_cpu<T>(h_bottom_grad.get(), h_top.get(), h_softmax_out.get(),
                              batch_size * head_num * seq_len_from, seq_len_to, scalar);

  ASSERT_TRUE(
      test::compare_array_approx<T>(d2h_bottom_grad.get(), h_bottom_grad.get(), tensor_size, eps));
}

}  // namespace

TEST(core23_masked_softmax_layer, fp32_16x2x16x16) {
  core23_masked_softmax_test<float>(16, 2, 16, 16, 0.25);
}
TEST(core23_masked_softmax_layer, fp32_512x4x64x128) {
  core23_masked_softmax_test<float>(512, 4, 64, 128, 0.884);
}
TEST(core23_masked_softmax_layer, fp32_1024x4x8x16) {
  core23_masked_softmax_test<float>(256, 4, 8, 16, 0.353);
}

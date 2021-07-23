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

#include "HugeCTR/include/layers/fused_reshape_concat_layer.hpp"

#include <gtest/gtest.h>
#include <utest/test_utils.h>

#include <vector>

#include "HugeCTR/include/utils.hpp"

using namespace std;
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

template <typename T>
void fused_reshape_concat_cpu(bool forward, T *output_item, T *output_ad, T **inputs,
                              size_t batch_size, size_t slot_num, int *vecs_size, int num,
                              size_t out_vector_size) {
  for (size_t l = 0; l < batch_size; l++) {
    for (size_t i = 0; i < slot_num; i++) {
      int count = 0;
      for (int j = 0; j < num; j++) {
        for (int k = 0; k < vecs_size[j]; k++) {
          if (forward) {
            if (i == slot_num - 1)
              output_ad[l * out_vector_size + count] =
                  inputs[j][l * vecs_size[j] * slot_num + i * vecs_size[j] + k];
            else
              output_item[l * (slot_num - 1) * out_vector_size + i * out_vector_size + count] =
                  inputs[j][l * vecs_size[j] * slot_num + i * vecs_size[j] + k];
          } else {
            if (i == slot_num - 1)
              inputs[j][l * vecs_size[j] * slot_num + i * vecs_size[j] + k] =
                  output_ad[l * out_vector_size + count];
            else
              inputs[j][l * vecs_size[j] * slot_num + i * vecs_size[j] + k] =
                  output_item[l * (slot_num - 1) * out_vector_size + i * out_vector_size + count];
          }
          count++;
        }
      }
    }
  }
}

template <typename T>
void fused_reshape_concat_test(size_t batch_size, size_t slot_num, std::vector<int> items) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  int num = items.size();
  size_t out_vector_size = 0;
  int *vecs_size = new int[num];
  Tensors2<T> in_tensors;

  for (int i = 0; i < num; i++) {
    size_t embedding_vec_size = items[i];
    vector<size_t> dims_in = {batch_size, slot_num, embedding_vec_size};
    Tensor2<T> in_tensor;
    buff->reserve(dims_in, &in_tensor);
    in_tensors.push_back(in_tensor);
    out_vector_size += embedding_vec_size;
    vecs_size[i] = embedding_vec_size;
  }

  Tensors2<T> out_tensors;
  size_t rows = batch_size * slot_num;
  size_t out_size_item = batch_size * (slot_num - 1) * out_vector_size;
  size_t out_size_ad = batch_size * out_vector_size;
  FusedReshapeConcatLayer<T> fused_reshape_concat_layer(in_tensors, out_tensors, buff,
                                                        test::get_default_gpu());

  buff->allocate();
  fused_reshape_concat_layer.initialize();
  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (int i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }

  T *d_out_item = out_tensors[0].get_ptr();
  T *d_out_ad = out_tensors[1].get_ptr();
  std::unique_ptr<T *[]> h_ins(new T *[num]);
  for (int i = 0; i < num; i++) {
    h_ins[i] = new T[rows * items[i]];
  }

  std::unique_ptr<T *[]> h_ins_b(new T *[num]);
  for (int i = 0; i < num; i++) {
    h_ins_b[i] = new T[rows * items[i]];
  }
  std::unique_ptr<T[]> h_out_item(new T[out_size_item]);
  std::unique_ptr<T[]> h_out_ad(new T[out_size_ad]);
  std::unique_ptr<T[]> h_cpu_out_item(new T[out_size_item]);
  std::unique_ptr<T[]> h_cpu_out_ad(new T[out_size_ad]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop
  for (int i = 0; i < num; i++) {
    size_t size = rows * items[i];
    simulator.fill(h_ins[i], size);
    CK_CUDA_THROW_(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  fused_reshape_concat_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(
      cudaMemcpy(h_out_item.get(), d_out_item, out_size_item * sizeof(T), cudaMemcpyDeviceToHost));
  CK_CUDA_THROW_(
      cudaMemcpy(h_out_ad.get(), d_out_ad, out_size_ad * sizeof(T), cudaMemcpyDeviceToHost));

  fused_reshape_concat_cpu(true, h_cpu_out_item.get(), h_cpu_out_ad.get(), h_ins.get(), batch_size,
                           slot_num, vecs_size, num, out_vector_size);
  ASSERT_TRUE(test::compare_array_approx<T>(h_out_ad.get(), h_cpu_out_ad.get(), out_size_ad,
                                            Eps<T>::value()));
  ASSERT_TRUE(test::compare_array_approx<T>(h_out_item.get(), h_cpu_out_item.get(), out_size_item,
                                            Eps<T>::value()));

  // bprop
  test::GaussianDataSimulator simulatorb(0.0f, 2.0f);
  for (int i = 0; i < num; i++) {
    size_t size = rows * items[i];
    memset(h_ins[i], 0, size * sizeof(T));
    CK_CUDA_THROW_(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }
  simulatorb.fill(h_out_item.get(), out_size_item);
  simulatorb.fill(h_out_ad.get(), out_size_ad);
  CK_CUDA_THROW_(
      cudaMemcpy(d_out_item, h_out_item.get(), out_size_item * sizeof(T), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(
      cudaMemcpy(d_out_ad, h_out_ad.get(), out_size_ad * sizeof(T), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  fused_reshape_concat_layer.bprop();  // compute wgrad and dgrad
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  for (int i = 0; i < num; i++) {
    size_t size = rows * items[i];
    CK_CUDA_THROW_(cudaMemcpy(h_ins_b[i], h_d_ins[i], size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  fused_reshape_concat_cpu(false, h_out_item.get(), h_out_ad.get(), h_ins.get(), batch_size,
                           slot_num, vecs_size, num, out_vector_size);
  for (int i = 0; i < num; i++) {
    size_t size = rows * items[i];
    ASSERT_TRUE(test::compare_array_approx<T>(h_ins[i], h_ins_b[i], size,
                                              Eps<T>::value()));  // compare dgrad
  }
}

}  // namespace

TEST(fused_reshape_concat_layer, fp32_32x20x12_3) {
  std::vector<int> items;
  int batch_size = 32, slot_num = 20;
  int goodID_size = 3, shopID_size = 5, cateID_size = 4;
  items.push_back(goodID_size);
  items.push_back(shopID_size);
  items.push_back(cateID_size);
  fused_reshape_concat_test<float>(batch_size, slot_num, items);
}

TEST(fused_reshape_concat_layer, fp32_32x20_7) {
  std::vector<int> items{21, 4, 7, 13, 75, 34, 13};
  fused_reshape_concat_test<float>(32, 20, items);
}

TEST(fused_reshape_concat_layer, fp32_32x100_16) {
  std::vector<int> items{21, 4, 7, 13, 75, 34, 13, 23, 76, 34, 13, 12, 14, 5, 8, 20};
  fused_reshape_concat_test<float>(32, 100, items);
}

TEST(fused_reshape_concat_layer, fp32_128x200_16) {
  std::vector<int> items{21, 54, 27, 13, 75, 34, 13, 23, 76, 34, 13, 12, 14, 5, 8, 20};
  fused_reshape_concat_test<float>(128, 200, items);
}

TEST(fused_reshape_concat_layer, fp32_128x1024_11) {
  std::vector<int> items{211, 54, 270, 130, 75, 34, 131, 231, 76, 341, 130};
  fused_reshape_concat_test<float>(128, 1024, items);
}
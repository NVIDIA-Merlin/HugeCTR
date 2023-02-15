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

#include <data_generator.hpp>
#include <general_buffer2.hpp>
#include <gpu_resource.hpp>
#include <inference/preallocated_buffer2.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-6f;

void preallocated_buffer2_test(int batch_size, int slot_num, int embedding_vec_size, int max_nnz) {
  CudaAllocator allocator;

  // row_ptrs: h_row_ptrs, d_row_ptrs, row_ptrs_tensor
  std::vector<size_t> row_ptrs_dims = {static_cast<size_t>(batch_size * slot_num + 1)};  // 1D
  size_t row_ptrs_size = 1;
  for (auto dim : row_ptrs_dims) {
    row_ptrs_size *= dim;
  }
  std::unique_ptr<int[]> h_row_ptrs(new int[row_ptrs_size]);
  std::shared_ptr<IDataSimulator<int>> ldata_sim;
  ldata_sim.reset(new IntUniformDataSimulator<int>(0, max_nnz));
  h_row_ptrs[0] = 0;
  for (size_t i = 1; i < row_ptrs_size; i++) {
    h_row_ptrs[i] = h_row_ptrs[i - 1] + ldata_sim->get_num();
  }

  size_t row_ptrs_size_in_bytes = row_ptrs_size * TensorScalarSizeFunc<int>::get_element_size();
  void* d_row_ptrs = allocator.allocate(row_ptrs_size_in_bytes);
  HCTR_LIB_THROW(
      cudaMemcpy(d_row_ptrs, h_row_ptrs.get(), row_ptrs_size_in_bytes, cudaMemcpyHostToDevice));
  std::shared_ptr<Tensor2<int>> row_ptrs_tensor = std::make_shared<Tensor2<int>>();

  HCTR_LOG(INFO, ROOT, "Bind the tensor to preallocated buffer for the first time\n");
  std::shared_ptr<TensorBuffer2> row_ptrs_buff =
      PreallocatedBuffer2<int>::create(d_row_ptrs, row_ptrs_dims);
  bind_tensor_to_buffer(row_ptrs_dims, row_ptrs_buff, row_ptrs_tensor);

  // embedding_features: h_embedding_features, d_embedding_features, embedding_features_tensor
  size_t feature_num = h_row_ptrs[row_ptrs_size - 1];
  std::vector<size_t> embedding_features_dims = {static_cast<size_t>(feature_num),
                                                 static_cast<size_t>(embedding_vec_size)};
  size_t embedding_features_size = 1;
  for (auto dim : embedding_features_dims) {
    embedding_features_size *= dim;
  }
  size_t embedding_features_size_in_bytes =
      embedding_features_size * TensorScalarSizeFunc<float>::get_element_size();
  std::unique_ptr<float[]> h_embedding_features(new float[embedding_features_size]);
  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_embedding_features.get(), embedding_features_size);
  void* d_embedding_features = allocator.allocate(embedding_features_size_in_bytes);
  HCTR_LIB_THROW(cudaMemcpy(d_embedding_features, h_embedding_features.get(),
                            embedding_features_size_in_bytes, cudaMemcpyHostToDevice));

  std::shared_ptr<Tensor2<float>> embedding_features_tensor = std::make_shared<Tensor2<float>>();
  std::shared_ptr<TensorBuffer2> embeddding_features_buff =
      PreallocatedBuffer2<float>::create(d_embedding_features, embedding_features_dims);
  bind_tensor_to_buffer(embedding_features_dims, embeddding_features_buff,
                        embedding_features_tensor);

  // copy Tensor2 back to cpu and compare with original buffer
  std::unique_ptr<int[]> h_row_ptrs_back(new int[row_ptrs_size]);
  std::unique_ptr<float[]> h_embedding_features_back(new float[embedding_features_size]);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaMemcpy(h_row_ptrs_back.get(), row_ptrs_tensor->get_ptr(),
                            row_ptrs_size_in_bytes, cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(cudaMemcpy(h_embedding_features_back.get(), embedding_features_tensor->get_ptr(),
                            embedding_features_size_in_bytes, cudaMemcpyDeviceToHost));
  ASSERT_TRUE(
      test::compare_array_approx<int>(h_row_ptrs.get(), h_row_ptrs_back.get(), row_ptrs_size, eps));
  ASSERT_TRUE(test::compare_array_approx<float>(
      h_embedding_features.get(), h_embedding_features_back.get(), embedding_features_size, eps));

  HCTR_LOG(INFO, ROOT, "Bind the tensor to preallocated buffer for the second time\n");
  void* d_row_ptrs2 = allocator.allocate(row_ptrs_size_in_bytes);
  HCTR_LIB_THROW(
      cudaMemcpy(d_row_ptrs2, h_row_ptrs.get(), row_ptrs_size_in_bytes, cudaMemcpyHostToDevice));

  std::shared_ptr<TensorBuffer2> row_ptrs_buff2 =
      PreallocatedBuffer2<int>::create(d_row_ptrs2, row_ptrs_dims);
  bind_tensor_to_buffer(row_ptrs_dims, row_ptrs_buff2, row_ptrs_tensor);

  // embedding_features: h_embedding_features, d_embedding_features, embedding_features_tensor
  void* d_embedding_features2 = allocator.allocate(embedding_features_size_in_bytes);
  HCTR_LIB_THROW(cudaMemcpy(d_embedding_features2, h_embedding_features.get(),
                            embedding_features_size_in_bytes, cudaMemcpyHostToDevice));

  std::shared_ptr<TensorBuffer2> embeddding_features_buff2 =
      PreallocatedBuffer2<float>::create(d_embedding_features2, embedding_features_dims);
  bind_tensor_to_buffer(embedding_features_dims, embeddding_features_buff2,
                        embedding_features_tensor);

  // copy Tensor2 back to cpu and compare with original buffer
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaMemcpy(h_row_ptrs_back.get(), row_ptrs_tensor->get_ptr(),
                            row_ptrs_size_in_bytes, cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(cudaMemcpy(h_embedding_features_back.get(), embedding_features_tensor->get_ptr(),
                            embedding_features_size_in_bytes, cudaMemcpyDeviceToHost));
  ASSERT_TRUE(
      test::compare_array_approx<int>(h_row_ptrs.get(), h_row_ptrs_back.get(), row_ptrs_size, eps));
  ASSERT_TRUE(test::compare_array_approx<float>(
      h_embedding_features.get(), h_embedding_features_back.get(), embedding_features_size, eps));
  // deallocate: d_row_ptrs2, d_embedding_features2
  allocator.deallocate(d_row_ptrs);
  allocator.deallocate(d_embedding_features);
  allocator.deallocate(d_row_ptrs2);
  allocator.deallocate(d_embedding_features2);
}

}  // namespace

TEST(preallocated_buffer2, fp32_10x1x64_10) { preallocated_buffer2_test(10, 1, 64, 10); }
TEST(preallocated_buffer2, fp32_10x10x64_1) { preallocated_buffer2_test(10, 10, 64, 1); }
TEST(preallocated_buffer2, fp32_4096x26x64_1) { preallocated_buffer2_test(4096, 26, 64, 1); }
TEST(preallocated_buffer2, fp32_4096x26x64_3) { preallocated_buffer2_test(4096, 26, 64, 3); }
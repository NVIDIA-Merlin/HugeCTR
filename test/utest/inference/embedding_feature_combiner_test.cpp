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

#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/inference/embedding_feature_combiner.hpp"
#include <vector>
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-2f;

template <typename TypeEmbedding>
void embedding_feature_combine_cpu(const float* input, TypeEmbedding* output, const int* row_ptrs,
                                  int batch_size, int slot_num, int embedding_vec_size, 
                                  EmbeddingFeatureCombiner_t combiner_type) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < slot_num; j++) {
      int feature_row_index = i * slot_num + j;
      int row_offset = row_ptrs[feature_row_index];                   // row offset within input
      int feature_num = row_ptrs[feature_row_index+1] - row_offset;  // num of feature vectors in one slot
      
      for (int k = 0; k < embedding_vec_size; k++) {
        float tmp = 0.0f;
        for (int l =0; l < feature_num; l++) {
          tmp += input[(row_offset + l)*embedding_vec_size + k];
        } // end for l
        if (combiner_type == EmbeddingFeatureCombiner_t::Mean)
          tmp /= feature_num;
        output[feature_row_index*embedding_vec_size + k] = tmp;
      } //end for k
    } // end for j
  } // end for i
}

template <>
void embedding_feature_combine_cpu(const float* input, __half* output, const int* row_ptrs,
                                  int batch_size, int slot_num, int embedding_vec_size, 
                                  EmbeddingFeatureCombiner_t combiner_type) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < slot_num; j++) {
      int feature_row_index = i * slot_num + j;
      int row_offset = row_ptrs[feature_row_index];                   // row offset within input
      int feature_num = row_ptrs[feature_row_index+1] - row_offset;  // num of feature vectors in one slot
      
      for (int k = 0; k < embedding_vec_size; k++) {
        float tmp = 0.0f;
        for (int l =0; l < feature_num; l++) {
          tmp += __half2float(input[(row_offset + l)*embedding_vec_size + k]);
        } // end for l
        if (combiner_type == EmbeddingFeatureCombiner_t::Mean && feature_num > 1) {         
          tmp /= feature_num;
        }
        output[feature_row_index*embedding_vec_size + k] = __float2half(tmp);
      } //end for k
    } // end for j
  } // end for i
}

template <typename TypeEmbedding>
void embedding_feature_combine_test(int batch_size, int slot_num, int embedding_vec_size,
                                   int max_nnz, EmbeddingFeatureCombiner_t combiner_type) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  
  Tensor2<int> row_ptrs_tensor;
  vector<size_t> row_ptrs_dims = { static_cast<size_t>(batch_size * slot_num + 1) };  // 1D
  buff->reserve(row_ptrs_dims, &row_ptrs_tensor);
  size_t row_ptrs_size = 1;
  for (auto dim : row_ptrs_dims) {
    row_ptrs_size *= dim;
  }
  std::unique_ptr<int[]> h_row_ptrs(new int[row_ptrs_size]);
  std::shared_ptr<IDataSimulator<int>> ldata_sim;
  ldata_sim.reset(new IntUniformDataSimulator<int>(0, max_nnz));
  h_row_ptrs[0] = 0;
  for (int i = 1; i < (int)row_ptrs_size; i++) {
    h_row_ptrs[i] = (h_row_ptrs[i-1] + ldata_sim->get_num());
  }

  size_t feature_num = h_row_ptrs[row_ptrs_size-1];
  Tensor2<float> in_tensor;
  vector<size_t> in_dims = {static_cast<size_t>(feature_num), static_cast<size_t>(embedding_vec_size)};  // 2D
  buff->reserve(in_dims, &in_tensor);
  
  Tensor2<TypeEmbedding> out_tensor;
  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  EmbeddingFeatureCombiner<TypeEmbedding> embedding_feature_combiner(in_tensor, row_ptrs_tensor, out_tensor,
                                                     batch_size, slot_num, combiner_type,
                                                     buff, test::get_default_gpu());
  buff->allocate();
  size_t in_size = 1;
  for (auto dim : in_dims) {
    in_size *= dim;
  }
  auto out_dims = out_tensor.get_dimensions();
  size_t out_size = 1;
  for (auto dim : out_dims) {
    out_size *= dim;
  }
  int* d_row_ptrs = row_ptrs_tensor.get_ptr();
  float* d_in = in_tensor.get_ptr();
  TypeEmbedding* d_out = out_tensor.get_ptr();
  std::unique_ptr<float[]> h_in(new float[in_size]);
  std::unique_ptr<TypeEmbedding[]> h_out(new TypeEmbedding[out_size]);
  std::unique_ptr<TypeEmbedding[]> h_cpu_out(new TypeEmbedding[out_size]);

  // fprop
  simulator.fill(h_in.get(), in_size);
  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), in_size * sizeof(float), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaMemcpy(d_row_ptrs, h_row_ptrs.get(), row_ptrs_size * sizeof(int), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  embedding_feature_combiner.fprop(false);
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), d_out, out_size * sizeof(TypeEmbedding), cudaMemcpyDeviceToHost));

  embedding_feature_combine_cpu(h_in.get(), h_cpu_out.get(), h_row_ptrs.get(), 
                               batch_size, slot_num, embedding_vec_size, combiner_type);
  ASSERT_TRUE(test::compare_array_approx<TypeEmbedding>(h_out.get(), h_cpu_out.get(), out_size, eps));

}

}  // namespace

TEST(embedding_feature_combiner, fp32_10x1x64_10_Sum) { embedding_feature_combine_test<float>(10, 1, 64, 10, EmbeddingFeatureCombiner_t::Sum); }
TEST(embedding_feature_combiner, fp32_10x10x64_1_Sum) { embedding_feature_combine_test<float>(10, 10, 64, 1, EmbeddingFeatureCombiner_t::Sum); }
TEST(embedding_feature_combiner, fp32_40960x26x64_1_Sum) { embedding_feature_combine_test<float>(40960, 26, 64, 1, EmbeddingFeatureCombiner_t::Sum); }
TEST(embedding_feature_combiner, fp32_40960x26x64_3_Sum) { embedding_feature_combine_test<float>(40960, 26, 64, 3, EmbeddingFeatureCombiner_t::Sum); }
TEST(embedding_feature_combiner, fp32_10x1x64_10_Mean) { embedding_feature_combine_test<float>(10, 1, 64, 10, EmbeddingFeatureCombiner_t::Mean); }
TEST(embedding_feature_combiner, fp32_10x10x64_1_Mean) { embedding_feature_combine_test<float>(10, 10, 64, 1, EmbeddingFeatureCombiner_t::Mean); }
TEST(embedding_feature_combiner, fp32_40960x26x64_1_Mean) { embedding_feature_combine_test<float>(40960, 26, 64, 1, EmbeddingFeatureCombiner_t::Mean); }
TEST(embedding_feature_combiner, fp32_40960x26x64_3_Mean) { embedding_feature_combine_test<float>(40960, 26, 64, 3, EmbeddingFeatureCombiner_t::Mean); }
TEST(embedding_feature_combiner, fp16_10x1x64_10_Sum) { embedding_feature_combine_test<__half>(10, 1, 64, 10, EmbeddingFeatureCombiner_t::Sum); }
TEST(embedding_feature_combiner, fp16_10x10x64_1_Sum) { embedding_feature_combine_test<__half>(10, 10, 64, 1, EmbeddingFeatureCombiner_t::Sum); }
TEST(embedding_feature_combiner, fp16_40960x26x64_1_Sum) { embedding_feature_combine_test<__half>(40960, 26, 64, 1, EmbeddingFeatureCombiner_t::Sum); }
TEST(embedding_feature_combiner, fp16_40960x26x64_3_Sum) { embedding_feature_combine_test<__half>(40960, 26, 64, 3, EmbeddingFeatureCombiner_t::Sum); }
TEST(embedding_feature_combiner, fp16_10x1x64_10_Mean) { embedding_feature_combine_test<__half>(10, 1, 64, 10, EmbeddingFeatureCombiner_t::Mean); }
TEST(embedding_feature_combiner, fp16_10x10x64_1_Mean) { embedding_feature_combine_test<__half>(10, 10, 64, 1, EmbeddingFeatureCombiner_t::Mean); }
TEST(embedding_feature_combiner, fp16_40960x26x64_1_Mean) { embedding_feature_combine_test<__half>(40960, 26, 64, 1, EmbeddingFeatureCombiner_t::Mean); }
TEST(embedding_feature_combiner, fp16_40960x26x64_3_Mean) { embedding_feature_combine_test<__half>(40960, 26, 64, 3, EmbeddingFeatureCombiner_t::Mean); }
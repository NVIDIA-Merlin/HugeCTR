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
#include "HugeCTR/include/inference/session_inference.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include <vector>
#include "gtest/gtest.h"
#include "utest/test_utils.h"
#include <cuda_profiler_api.h>

using namespace HugeCTR;
namespace {

void session_inference_test(int batch_size, int slot_num, int embedding_vec_size,
                            int num_samples, int max_nnz, int max_feature_num_per_sample, int dense_dim,
                            EmbeddingFeatureCombiner_t combiner_type,
                            const std::string& config_file_path) {
  
  CudaAllocator allocator;

  std::vector<size_t> row_ptrs_dims = { static_cast<size_t>(batch_size * slot_num + 1) };  // 1D
  size_t row_ptrs_size = 1;
  for (auto dim : row_ptrs_dims) {
    row_ptrs_size *= dim;
  }
  std::unique_ptr<int[]> h_row_ptrs(new int[row_ptrs_size]);
  std::shared_ptr<IDataSimulator<int>> ldata_sim;
  ldata_sim.reset(new IntUniformDataSimulator<int>(1, max_nnz));
  h_row_ptrs[0] = 0;
  for (int i = 1; i < (int)row_ptrs_size; i++) {
    h_row_ptrs[i] = (h_row_ptrs[i-1] + ldata_sim->get_num());
  }
  
  std::cout << "batch_size: " << batch_size << ", slot_num: " << slot_num << ", embedding_vec_size: " << embedding_vec_size 
            << ", max_nnz:" << max_nnz << ", num_samples: " << num_samples << std::endl;

  std::cout << "==========================row offset ptrs===================" << std::endl;
  for (int i = 0; i < (int)row_ptrs_size; i++) {
    std::cout << h_row_ptrs[i] << " ";
  }
  std::cout << std::endl;
  
  
  size_t row_ptrs_size_in_bytes = row_ptrs_size * TensorScalarSizeFunc<int>::get_element_size();
  void* d_row_ptrs = allocator.allocate(row_ptrs_size_in_bytes);
  CK_CUDA_THROW_(cudaMemcpy(d_row_ptrs, h_row_ptrs.get(), row_ptrs_size_in_bytes, cudaMemcpyHostToDevice));
  
  std::vector<size_t> embedding_features_dims = {static_cast<size_t>(max_feature_num_per_sample), static_cast<size_t>(embedding_vec_size)};
  size_t embedding_features_size = 1;
  for (auto dim : embedding_features_dims) {
    embedding_features_size *= dim;
  }
  size_t embedding_features_size_in_bytes = embedding_features_size * TensorScalarSizeFunc<float>::get_element_size();
  std::unique_ptr<float[]> h_embedding_features(new float[embedding_features_size]);
  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_embedding_features.get(), embedding_features_size);
  
  std::cout << "==========================embedding features===================" << std::endl;
  for (int i = 0; i < h_row_ptrs[row_ptrs_size-1]; i++) {
    for (int j = 0; j < embedding_vec_size; j++) {
        std::cout << h_embedding_features[i*embedding_vec_size + j] <<" ";
    }
    std::cout << std::endl;
  }
  
  void* d_embedding_features = allocator.allocate(embedding_features_size_in_bytes);
  CK_CUDA_THROW_(cudaMemcpy(d_embedding_features, h_embedding_features.get(), embedding_features_size_in_bytes, cudaMemcpyHostToDevice));

  float* dense = reinterpret_cast<float*>(allocator.allocate(batch_size*dense_dim*sizeof(float)));
  float* output = reinterpret_cast<float*>(allocator.allocate(batch_size*sizeof(float)));
  int* row_ptrs = reinterpret_cast<int*>(d_row_ptrs);
  float* embedding_features = reinterpret_cast<float*>(d_embedding_features);

  InferenceSession sess(config_file_path, 0);
  sess.predict(dense, row_ptrs, embedding_features, output, num_samples);
  
  std::unique_ptr<float[]> h_out(new float[batch_size]);
  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), output, batch_size*sizeof(float), cudaMemcpyDeviceToHost));
  

  std::cout << "==========================prediction result===================" << std::endl;
  for (int i = 0; i < batch_size; i++) {
      std::cout << h_out[i] << " ";
  }
  std::cout << std::endl;

  allocator.deallocate(d_row_ptrs);
  allocator.deallocate(d_embedding_features);
  allocator.deallocate(dense);
  allocator.deallocate(output);
}

}  // namespace

//TEST(inference_session_test, inference_parser_test) {
//  std::string json_name = PROJECT_HOME_ + "utest/simple_inference_config.json";
//  test_inference_session(json_name);
//}

TEST(session_inference, fp32_2x2x10x1x3_10_Sum) { session_inference_test(2, 2, 10, 1, 3, 10, 10, EmbeddingFeatureCombiner_t::Sum, "/hugectr/test/utest/simple_inference_config.json"); }
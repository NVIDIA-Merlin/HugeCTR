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
#include "HugeCTR/include/inference/embedding_interface.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include <vector>
#include "gtest/gtest.h"
#include "utest/test_utils.h"
#include <cuda_profiler_api.h>

using namespace HugeCTR;
namespace {

struct InferenceParams {
  int max_batchsize;
  int dense_dim;           
  std::vector<int> slot_num;
  std::vector<int> max_feature_num_per_sample;
  std::vector<int> embedding_vec_size;
  std::vector<EmbeddingFeatureCombiner_t> combiner_type;
  InferenceParams(const nlohmann::json& config);
};

InferenceParams::InferenceParams(const nlohmann::json& config) {
  auto j = get_json(config, "inference");
  max_batchsize = get_value_from_json<int>(j, "max_batchsize");
  auto j_layers_array = get_json(config, "layers");
  const nlohmann::json& j_data = j_layers_array[0];
  auto j_dense = get_json(j_data, "dense");
  dense_dim = get_value_from_json<int>(j_dense, "dense_dim");
  auto j_sparse_inputs = get_json(j_data, "sparse");

  for (unsigned int i = 0; i < j_sparse_inputs.size(); i++) {
    const nlohmann::json& j_sparse = j_sparse_inputs[0];
    slot_num.push_back(get_value_from_json<int>(j_sparse, "slot_num"));
    max_feature_num_per_sample.push_back(get_value_from_json<int>(j_sparse, "max_feature_num_per_sample"));
  }
  // get embedding params: embedding_vec_size, combiner_type
  {
    for (unsigned int i = 1; i < j_layers_array.size(); i++) {
      // if not embedding then break
      const nlohmann::json& j = j_layers_array[i];
      auto embedding_name = get_value_from_json<std::string>(j, "type");
      if (embedding_name != "DistributedSlotSparseEmbeddingHash" &&
          embedding_name != "LocalizedSlotSparseEmbeddingHash" &&
          embedding_name != "LocalizedSlotSparseEmbeddingOneHot") {
        break;
      }
      auto j_embed_params =  get_json(j, "sparse_embedding_hparam");
      auto vec_size = get_value_from_json<int>(j_embed_params, "embedding_vec_size");
      auto combiner = get_value_from_json<int>(j_embed_params, "combiner");
      embedding_vec_size.push_back(vec_size);
      if (combiner == 1) {
        combiner_type.push_back(EmbeddingFeatureCombiner_t::Mean);
      } else {
        combiner_type.push_back(EmbeddingFeatureCombiner_t::Sum);
      }
    }  // for ()
  }    // get embedding params
}
void session_inference_test(const std::string& config_file, int num_samples) {
  InferenceParams inference_params(read_json_file(config_file));
  int batch_size = inference_params.max_batchsize;
  int dense_dim = inference_params.dense_dim;
  int slot_num = inference_params.slot_num[0];
  int max_feature_num_per_sample = inference_params.max_feature_num_per_sample[0];
  int embedding_vec_size = inference_params.embedding_vec_size[0];
  int max_nnz = max_feature_num_per_sample / slot_num;
  num_samples = num_samples < batch_size? num_samples:batch_size;
  CudaAllocator allocator;
  CudaHostAllocator host_allocator;
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
            << ", max_nnz:" << max_nnz << ", num_samples: " << num_samples << ", max_feature_num_per_sample: " << max_feature_num_per_sample 
            << std::endl;

  size_t row_ptrs_size_in_bytes = row_ptrs_size * TensorScalarSizeFunc<int>::get_element_size();
  void* d_row_ptrs = allocator.allocate(row_ptrs_size_in_bytes);
  CK_CUDA_THROW_(cudaMemcpy(d_row_ptrs, h_row_ptrs.get(), row_ptrs_size_in_bytes, cudaMemcpyHostToDevice));
  
  max_feature_num_per_sample = max_feature_num_per_sample < slot_num*max_nnz? max_feature_num_per_sample:slot_num*max_nnz;
  std::vector<size_t> embedding_features_dims = {static_cast<size_t>(batch_size), static_cast<size_t>(max_feature_num_per_sample), static_cast<size_t>(embedding_vec_size)};
  size_t embedding_features_size = 1;
  for (auto dim : embedding_features_dims) {
    embedding_features_size *= dim;
  }

  size_t embedding_features_size_in_bytes = embedding_features_size * TensorScalarSizeFunc<float>::get_element_size();
  std::unique_ptr<float[]> h_embedding_features(new float[embedding_features_size]);
  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_embedding_features.get(), embedding_features_size);

  void* d_embedding_features = allocator.allocate(embedding_features_size_in_bytes);
  CK_CUDA_THROW_(cudaMemcpy(d_embedding_features, h_embedding_features.get(), embedding_features_size_in_bytes, cudaMemcpyHostToDevice));

  size_t dense_size = batch_size * dense_dim;
  std::unique_ptr<float[]> h_dense(new float[dense_size]);
  FloatUniformDataSimulator<float> fdata_sim(0, 1);     
  for (int i = 0; i < (int)dense_size; i++)
    h_dense[i] = fdata_sim.get_num();

  size_t dense_size_in_bytes = dense_size * TensorScalarSizeFunc<float>::get_element_size();
  float* dense = reinterpret_cast<float*>(allocator.allocate(batch_size*dense_dim*sizeof(float)));
  float* output = reinterpret_cast<float*>(allocator.allocate(batch_size*sizeof(float)));
  void* h_embeddingcolumns = host_allocator.allocate(batch_size*max_feature_num_per_sample*sizeof(int));
  int* row_ptrs = reinterpret_cast<int*>(d_row_ptrs);
  float* embedding_features = reinterpret_cast<float*>(d_embedding_features);
  std::unique_ptr<float[]> h_out(new float[batch_size]);
  embedding_interface* embedding_ptr = nullptr;
  InferenceSession sess(config_file, 0, embedding_ptr);

  CK_CUDA_THROW_(cudaMemcpy(dense, h_dense.get(), dense_size_in_bytes, cudaMemcpyHostToDevice));  
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  sess.predict(dense, h_embeddingcolumns, row_ptrs, embedding_features, num_samples);  //fake
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), output, batch_size*sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << "==========================prediction result===================" << std::endl;
  for (int i = 0; i < batch_size; i++) {
      std::cout << h_out[i] << " ";
  }
  std::cout << std::endl;

  host_allocator.deallocate(h_embeddingcolumns);
  allocator.deallocate(d_row_ptrs);
  allocator.deallocate(d_embedding_features);
  allocator.deallocate(dense);
  allocator.deallocate(output);
}

}  // namespace

TEST(session_inference, fp32_1x26x16x1x1_30_Sum) { session_inference_test("/hugectr_ci_workdir/test/utest/simple_inference_config.json", 1); }
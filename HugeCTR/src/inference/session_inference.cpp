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

#include "HugeCTR/include/inference/session_inference.hpp"

#include <iostream>
#include <vector>
namespace HugeCTR {

InferenceSession::InferenceSession(const std::string& config_file, int device_id, embedding_interface* embedding_ptr)
    : config_(read_json_file(config_file)),
      parser_(config_),
      inference_parser_(config_),
      embedding_table_slot_size({0}),
      resource_manager_(ResourceManager::create({{device_id}}, 0)) {
  try {
    Network* network_ptr;
    parser_.create_pipeline(inference_parser_, dense_input_tensor_, row_ptrs_tensors_, embedding_features_tensors_, embedding_table_slot_size, &embedding_feature_combiners_, &network_ptr,  resource_manager_);
    network_ = std::move(std::unique_ptr<Network>(network_ptr));
    if(inference_parser_.dense_model_file.size() > 0) {
      network_->upload_params_to_device(inference_parser_.dense_model_file);
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  return;
}  // namespace HugeCTR

InferenceSession::~InferenceSession() {}

void InferenceSession::predict(float* d_dense, void* h_embeddingcolumns, int *d_row_ptrs, float* d_output, int num_samples) {
  float* d_embeddingvectors = d_output; // fake

  size_t embedding_table_num = embedding_feature_combiners_.size();
  if (embedding_table_num !=  row_ptrs_tensors_.size() || 
      embedding_table_num != embedding_features_tensors_.size() ||
      embedding_table_num <= 0) {
    CK_THROW_(Error_t::IllegalCall, "embedding feature combiner inconsistent");
  }

  auto dense_dims = dense_input_tensor_.get_dimensions();
  size_t dense_size = 1;
  for (auto dim : dense_dims) {
    dense_size *= dim;
  }
  size_t dense_size_in_bytes = dense_size * sizeof(float);
  CK_CUDA_THROW_(cudaMemcpy(dense_input_tensor_.get_ptr(), d_dense, dense_size_in_bytes, cudaMemcpyDeviceToDevice));
  
  auto row_ptrs_dims = row_ptrs_tensors_[0]->get_dimensions();
  std::shared_ptr<TensorBuffer2> row_ptrs_buff = PreallocatedBuffer2<int>::create(d_row_ptrs, row_ptrs_dims);
  bind_tensor_to_buffer(row_ptrs_dims, row_ptrs_buff, row_ptrs_tensors_[0]);

  // to be done
  //embeddingcache->look_up(d_embeddingcolumns, d_embeddingvectors, ...);

  auto embedding_features_dims = embedding_features_tensors_[0]->get_dimensions();
  std::shared_ptr<TensorBuffer2> embeddding_features_buff = PreallocatedBuffer2<float>::create(d_embeddingvectors, embedding_features_dims);
  bind_tensor_to_buffer(embedding_features_dims, embeddding_features_buff, embedding_features_tensors_[0]);

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  embedding_feature_combiners_[0]->fprop(false);
  network_->predict();
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  float* d_pred = network_->get_pred_tensor().get_ptr();
  CK_CUDA_THROW_(cudaMemcpy(d_output, d_pred, inference_parser_.max_batchsize*sizeof(float), cudaMemcpyDeviceToDevice));
}

}  // namespace HugeCTR

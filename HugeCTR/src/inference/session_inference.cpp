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

InferenceSession::InferenceSession(const std::string& model_config_path, const InferenceParams& inference_params, const std::shared_ptr<embedding_interface>& embedding_cache)
    : config_(read_json_file(model_config_path)),
      embedding_table_slot_size_({0}),
      resource_manager_(ResourceManager::create({{inference_params.device_id}}, 0)),
      embedding_cache_(embedding_cache),
      inference_parser_(config_),
      inference_params_(inference_params) {
  try {
    Network* network_ptr;
    inference_parser_.create_pipeline(inference_params_, dense_input_tensor_, row_ptrs_tensors_, embedding_features_tensors_, embedding_table_slot_size_, &embedding_feature_combiners_, &network_ptr,  resource_manager_);
    network_ = std::move(std::unique_ptr<Network>(network_ptr));
    network_->initialize(false);
    if(inference_params_.dense_model_file.size() > 0) {
      network_->upload_params_to_device_inference(inference_params_.dense_model_file);
    }
    CudaDeviceContext ctx;
    ctx.set_device(inference_params.device_id);
    for(unsigned int idx_embedding_table = 1; idx_embedding_table < embedding_table_slot_size_.size(); ++idx_embedding_table){
      cudaStream_t lookup_stream;
      cudaStreamCreateWithFlags(&lookup_stream, cudaStreamNonBlocking);
      lookup_streams_.push_back(lookup_stream);
      cudaStream_t update_stream;
      cudaStreamCreateWithFlags(&update_stream, cudaStreamNonBlocking);
      update_streams_.push_back(update_stream);
    }
    workspace_handler_ = embedding_cache_->create_workspace();
    CK_CUDA_THROW_(cudaMalloc((void**)&d_embeddingvectors_, inference_params_.max_batchsize *  inference_parser_.max_embedding_vector_size_per_sample * sizeof(float)));
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  return;
}  // namespace HugeCTR

InferenceSession::~InferenceSession() {
  embedding_cache_->destroy_workspace(workspace_handler_);
  cudaFree(d_embeddingvectors_);
  for (auto stream : lookup_streams_)
    cudaStreamDestroy(stream);
  for (auto stream : update_streams_)
    cudaStreamDestroy(stream);
}

void InferenceSession::separate_keys_by_table_(int* d_row_ptrs, const std::vector<size_t>& embedding_table_slot_size, int num_samples) {
  size_t slot_num = inference_parser_.slot_num;
  size_t num_embedding_tables = inference_parser_.num_embedding_tables;
  size_t row_ptrs_size_sample = num_samples * slot_num + 1;
  size_t row_ptrs_size_in_bytes_sample = row_ptrs_size_sample * sizeof(int);
  std::unique_ptr<int[]> h_row_ptrs(new int[row_ptrs_size_sample]);
  CK_CUDA_THROW_(cudaMemcpy(h_row_ptrs.get(), d_row_ptrs, row_ptrs_size_in_bytes_sample, cudaMemcpyDeviceToHost));
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  h_embedding_offset_.resize(num_samples*num_embedding_tables+1);
  for (int i = 0; i < num_samples; i++) {
    for (int j = 0; j < static_cast<int>(num_embedding_tables); j++) {
      h_embedding_offset_[i*num_embedding_tables + j + 1] = h_row_ptrs[i*slot_num + static_cast<int>(embedding_table_slot_size[j+1])];
    }
  }
}

void InferenceSession::predict(float* d_dense, void* h_embeddingcolumns, int *d_row_ptrs, float* d_output, int num_samples) {
  size_t num_embedding_tables = inference_parser_.num_embedding_tables;
  if (num_embedding_tables !=  row_ptrs_tensors_.size() || 
      num_embedding_tables != embedding_features_tensors_.size() ||
      num_embedding_tables != embedding_feature_combiners_.size()) {
    CK_THROW_(Error_t::IllegalCall, "embedding feature combiner inconsistent");
  }
  // embedding cache look up and update
  separate_keys_by_table_(d_row_ptrs, embedding_table_slot_size_, num_samples);
  embedding_cache_->look_up(h_embeddingcolumns, h_embedding_offset_, d_embeddingvectors_, workspace_handler_, lookup_streams_);
  CK_CUDA_THROW_(cudaStreamSynchronize(lookup_streams_[0]));
  if (workspace_handler_.use_gpu_embedding_cache_ &&
        workspace_handler_.h_hit_rate_[0] < inference_params_.hit_rate_threshold) {
    embedding_cache_->update(workspace_handler_, lookup_streams_);
  }
  CK_CUDA_THROW_(cudaStreamSynchronize(update_streams_[0]));

  // copy dense input to dense tensor
  auto dense_dims = dense_input_tensor_.get_dimensions();
  size_t dense_size = 1;
  for (auto dim : dense_dims) {
    dense_size *= dim;
  }
  size_t dense_size_in_bytes = dense_size * sizeof(float);
  CK_CUDA_THROW_(cudaMemcpyAsync(dense_input_tensor_.get_ptr(), d_dense, dense_size_in_bytes, cudaMemcpyDeviceToDevice, resource_manager_->get_local_gpu(0)->get_stream()));
  
  // bind row ptrs input to row ptrs tensor 
  auto row_ptrs_dims = row_ptrs_tensors_[0]->get_dimensions();
  std::shared_ptr<TensorBuffer2> row_ptrs_buff = PreallocatedBuffer2<int>::create(d_row_ptrs, row_ptrs_dims);
  bind_tensor_to_buffer(row_ptrs_dims, row_ptrs_buff, row_ptrs_tensors_[0]);

  // bind embedding vectors from looking up to embedding features tensor 
  auto embedding_features_dims = embedding_features_tensors_[0]->get_dimensions();
  std::shared_ptr<TensorBuffer2> embeddding_features_buff = PreallocatedBuffer2<float>::create(d_embeddingvectors_, embedding_features_dims);
  bind_tensor_to_buffer(embedding_features_dims, embeddding_features_buff, embedding_features_tensors_[0]);
  
  // feature combiner & dense network feedforward, they are both using resource_manager_->get_local_gpu(0)->get_stream()
  embedding_feature_combiners_[0]->fprop(false);
  network_->predict();
  
  // copy the prediction result to output
  float* d_pred = network_->get_pred_tensor().get_ptr();
  CK_CUDA_THROW_(cudaMemcpyAsync(d_output, d_pred, inference_params_.max_batchsize*sizeof(float), cudaMemcpyDeviceToDevice, resource_manager_->get_local_gpu(0)->get_stream()));
  CK_CUDA_THROW_(cudaStreamSynchronize(resource_manager_->get_local_gpu(0)->get_stream()));
}

}  // namespace HugeCTR

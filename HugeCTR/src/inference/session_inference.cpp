/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <HugeCTR/include/inference/session_inference.hpp>
#include <HugeCTR/include/resource_managers/resource_manager_ext.hpp>
#include <HugeCTR/include/utils.hpp>
#include <iostream>
#include <vector>
namespace HugeCTR {

InferenceSession::InferenceSession(const std::string& model_config_path,
                                   const InferenceParams& inference_params,
                                   const std::shared_ptr<embedding_interface>& embedding_cache)
    : config_(read_json_file(model_config_path)),
      embedding_table_slot_size_({0}),
      embedding_cache_(embedding_cache),
      inference_parser_(config_),
      inference_params_(inference_params),
      resource_manager_(ResourceManagerCore::create({{inference_params.device_id}}, 0)) {
  try {
    if (inference_params_.use_gpu_embedding_cache &&
        embedding_cache->get_device_id() != inference_params_.device_id) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "The device id of inference_params is not consistent with that of embedding cache.");
    }
    auto b2s = [](const char val) { return val ? "True" : "False"; };
    HCTR_LOG(INFO, ROOT, "Model name: %s\n", inference_params_.model_name.c_str());
    HCTR_LOG(INFO, ROOT, "Use mixed precision: %s\n", b2s(inference_params.use_mixed_precision));
    HCTR_LOG(INFO, ROOT, "Use cuda graph: %s\n", b2s(inference_params.use_cuda_graph));
    HCTR_LOG(INFO, ROOT, "Max batchsize: %lu\n", inference_params.max_batchsize);
    HCTR_LOG(INFO, ROOT, "Use I64 input key: %s\n", b2s(inference_params.i64_input_key));
    Network* network_ptr;
    inference_parser_.create_pipeline(
        inference_params_, dense_input_tensorbag_, row_ptrs_tensors_, embedding_features_tensors_,
        embedding_table_slot_size_, &embedding_feature_combiners_, &network_ptr, resource_manager_);
    network_ = std::move(std::unique_ptr<Network>(network_ptr));
    network_->initialize(false);
    if (inference_params.use_algorithm_search) {
      network_->search_algorithm();
    }
    if (inference_params_.dense_model_file.size() > 0) {
      network_->upload_params_to_device_inference(inference_params_.dense_model_file);
    }
    CudaDeviceContext context(inference_params_.device_id);
    for (unsigned int idx_embedding_table = 1;
         idx_embedding_table < embedding_table_slot_size_.size(); ++idx_embedding_table) {
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      streams_.push_back(stream);
    }
    HCTR_LIB_THROW(cudaMalloc((void**)&d_embeddingvectors_,
                              inference_params_.max_batchsize *
                                  inference_parser_.max_embedding_vector_size_per_sample *
                                  sizeof(float)));
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
  return;
}  // namespace HugeCTR

InferenceSession::~InferenceSession() {
  CudaDeviceContext context(inference_params_.device_id);
  cudaFree(d_embeddingvectors_);
  for (auto stream : streams_) cudaStreamDestroy(stream);
}

void InferenceSession::separate_keys_by_table_(int* d_row_ptrs,
                                               const std::vector<size_t>& embedding_table_slot_size,
                                               int num_samples) {
  size_t slot_num = inference_parser_.slot_num;
  size_t num_embedding_tables = inference_parser_.num_embedding_tables;
  size_t row_ptrs_size_sample = num_samples * slot_num + num_embedding_tables;
  size_t row_ptrs_size_in_bytes_sample = row_ptrs_size_sample * sizeof(int);
  std::unique_ptr<int[]> h_row_ptrs(new int[row_ptrs_size_sample]);
  HCTR_LIB_THROW(cudaMemcpyAsync(h_row_ptrs.get(), d_row_ptrs, row_ptrs_size_in_bytes_sample,
                                 cudaMemcpyDeviceToHost,
                                 resource_manager_->get_local_gpu(0)->get_stream()));
  HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(0)->get_stream()));
  h_embedding_offset_.resize(num_samples * num_embedding_tables + 1);

  for (int i = 0; i < num_samples; i++) {
    size_t acc_emb_key_offset = 0;
    for (int j = 0; j < static_cast<int>(num_embedding_tables); j++) {
      size_t num_of_feature =
          h_row_ptrs[(i + 1) * inference_parser_.slot_num_for_tables[j] + acc_emb_key_offset] -
          h_row_ptrs[i * inference_parser_.slot_num_for_tables[j] + acc_emb_key_offset];
      h_embedding_offset_[i * num_embedding_tables + j + 1] =
          h_embedding_offset_[i * num_embedding_tables + j] + num_of_feature;
      acc_emb_key_offset += num_samples * inference_parser_.slot_num_for_tables[j] + 1;
    }
  }
}

void InferenceSession::predict(float* d_dense, void* h_embeddingcolumns, int* d_row_ptrs,
                               float* d_output, int num_samples) {
  size_t num_embedding_tables = inference_parser_.num_embedding_tables;
  if (num_embedding_tables != row_ptrs_tensors_.size() ||
      num_embedding_tables != embedding_features_tensors_.size() ||
      num_embedding_tables != embedding_feature_combiners_.size()) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "embedding feature combiner inconsistent");
  }
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());

  // apply the memory block for embedding cache workspace
  memory_block_ = NULL;
  while (memory_block_ == NULL) {
    memory_block_ = reinterpret_cast<struct MemoryBlock*>(embedding_cache_->get_worker_space(
        inference_params_.model_name, inference_params_.device_id, CACHE_SPACE_TYPE::WORKER));
  }
  // embedding cache look up and update
  separate_keys_by_table_(d_row_ptrs, embedding_table_slot_size_, num_samples);
  bool sync_flag =
      embedding_cache_->look_up(h_embeddingcolumns, h_embedding_offset_, d_embeddingvectors_,
                                memory_block_, streams_, inference_params_.hit_rate_threshold);
  // free the memory block for the current embedding cache
  if (sync_flag) {
    embedding_cache_->free_worker_space(memory_block_);
  }

  // copy dense input to dense tensor
  auto dense_dims = dense_input_tensorbag_.get_dimensions();
  size_t dense_size = 1;
  for (auto dim : dense_dims) {
    dense_size *= dim;
  }

  if (inference_params_.use_mixed_precision) {
    convert_array_on_device(reinterpret_cast<__half*>(dense_input_tensorbag_.get_ptr()), d_dense,
                            dense_size, resource_manager_->get_local_gpu(0)->get_stream());
  } else {
    convert_array_on_device(reinterpret_cast<float*>(dense_input_tensorbag_.get_ptr()), d_dense,
                            dense_size, resource_manager_->get_local_gpu(0)->get_stream());
  }

  size_t acc_emb_table_offset = 0;
  size_t acc_emb_row_offset = 0;
  for (int j = 0; j < static_cast<int>(num_embedding_tables); j++) {
    // bind row ptrs input to row ptrs tensor
    auto row_ptrs_dims = row_ptrs_tensors_[j]->get_dimensions();
    std::shared_ptr<TensorBuffer2> row_ptrs_buff =
        PreallocatedBuffer2<int>::create(d_row_ptrs + acc_emb_row_offset, row_ptrs_dims);
    bind_tensor_to_buffer(row_ptrs_dims, row_ptrs_buff, row_ptrs_tensors_[j]);
    acc_emb_row_offset += num_samples * inference_parser_.slot_num_for_tables[j] + 1;

    // bind embedding vectors from looking up to embedding features tensor
    auto embedding_features_dims = embedding_features_tensors_[j]->get_dimensions();
    std::shared_ptr<TensorBuffer2> embeddding_features_buff = PreallocatedBuffer2<float>::create(
        d_embeddingvectors_ + acc_emb_table_offset, embedding_features_dims);
    bind_tensor_to_buffer(embedding_features_dims, embeddding_features_buff,
                          embedding_features_tensors_[j]);
    acc_emb_table_offset += inference_params_.max_batchsize *
                            inference_parser_.max_feature_num_for_tables[j] *
                            inference_parser_.embed_vec_size_for_tables[j];
    // feature combiner & dense network feedforward, they are both using
    // resource_manager_->get_local_gpu(0)->get_stream()
    embedding_feature_combiners_[j]->fprop(false);
  }
  network_->predict();

  // copy the prediction result to output
  if (inference_params_.use_mixed_precision) {
    convert_array_on_device(d_output, network_->get_pred_tensor_half().get_ptr(),
                            network_->get_pred_tensor_half().get_num_elements(),
                            resource_manager_->get_local_gpu(0)->get_stream());
  } else {
    convert_array_on_device(d_output, network_->get_pred_tensor().get_ptr(),
                            network_->get_pred_tensor().get_num_elements(),
                            resource_manager_->get_local_gpu(0)->get_stream());
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(0)->get_stream()));
}

}  // namespace HugeCTR

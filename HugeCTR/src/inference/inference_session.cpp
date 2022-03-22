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

#include <inference/inference_session.hpp>
#include <iostream>
#include <resource_managers/resource_manager_core.hpp>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

InferenceSessionBase::~InferenceSessionBase() = default;

std::shared_ptr<InferenceSessionBase> InferenceSessionBase::create(
    const std::string& model_config_path, const InferenceParams& inference_params,
    const std::shared_ptr<EmbeddingCacheBase>& embedding_cache) {
  return std::make_shared<InferenceSession>(model_config_path, inference_params, embedding_cache);
}

InferenceSession::InferenceSession(const std::string& model_config_path,
                                   const InferenceParams& inference_params,
                                   const std::shared_ptr<EmbeddingCacheBase>& embedding_cache)
    : InferenceSessionBase(),
      config_(read_json_file(model_config_path)),
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
    HCTR_LOG(INFO, ROOT, "Create inference session on device: %d\n", inference_params_.device_id);
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
    for (size_t idx = 0; idx < inference_params_.sparse_model_files.size(); ++idx) {
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      streams_.push_back(stream);
    }
    h_row_ptrs_ = (int*)malloc((inference_params_.max_batchsize * inference_parser_.slot_num +
                                inference_parser_.num_embedding_tables) *
                               sizeof(int));
    // h_keys_ is a void pointer, which serves key types of both long long and unsigned int
    h_keys_ = malloc(inference_params_.max_batchsize *
                     inference_parser_.max_feature_num_per_sample * sizeof(long long));
    HCTR_LIB_THROW(cudaMalloc((void**)&d_embedding_vectors_,
                              inference_params_.max_batchsize *
                                  inference_parser_.max_embedding_vector_size_per_sample *
                                  sizeof(float)));
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
  return;
}

InferenceSession::~InferenceSession() {
  CudaDeviceContext context(inference_params_.device_id);
  cudaFree(d_embedding_vectors_);
  free(h_keys_);
  free(h_row_ptrs_);
  for (auto stream : streams_) cudaStreamDestroy(stream);
}

void InferenceSession::predict(float* d_dense, void* h_embeddingcolumns, int* d_row_ptrs,
                               float* d_output, int num_samples) {
  size_t num_embedding_tables = inference_parser_.num_embedding_tables;
  if (num_embedding_tables != row_ptrs_tensors_.size() ||
      num_embedding_tables != embedding_features_tensors_.size() ||
      num_embedding_tables != embedding_feature_combiners_.size()) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "embedding feature combiner inconsistent");
  }
  // Copy row_ptrs to host
  HCTR_LIB_THROW(cudaMemcpy(
      h_row_ptrs_, d_row_ptrs,
      (num_samples * inference_parser_.slot_num + inference_parser_.num_embedding_tables) *
          sizeof(int),
      cudaMemcpyDeviceToHost));

  // Redistribute keys ：from sample first to table first
  if (inference_params_.i64_input_key) {
    distribute_keys_per_table(static_cast<long long*>(h_keys_),
                              static_cast<long long*>(h_embeddingcolumns), h_row_ptrs_, num_samples,
                              inference_parser_.slot_num_for_tables);
  } else {
    distribute_keys_per_table(static_cast<unsigned int*>(h_keys_),
                              static_cast<unsigned int*>(h_embeddingcolumns), h_row_ptrs_,
                              num_samples, inference_parser_.slot_num_for_tables);
  }

  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  // embedding_cache lookup
  size_t acc_vectors_offset{0};
  size_t acc_row_ptrs_offset{0};
  size_t acc_keys_offset{0};
  size_t num_keys{0};
  for (size_t i = 0; i < num_embedding_tables; ++i) {
    acc_row_ptrs_offset += num_samples * inference_parser_.slot_num_for_tables[i] + 1;
    num_keys = h_row_ptrs_[acc_row_ptrs_offset - 1];
    if (inference_params_.i64_input_key) {
      embedding_cache_->lookup(i, d_embedding_vectors_ + acc_vectors_offset,
                               static_cast<const long long*>(h_keys_) + acc_keys_offset, num_keys,
                               inference_params_.hit_rate_threshold, streams_[i]);
    } else {
      embedding_cache_->lookup(i, d_embedding_vectors_ + acc_vectors_offset,
                               static_cast<const unsigned int*>(h_keys_) + acc_keys_offset,
                               num_keys, inference_params_.hit_rate_threshold, streams_[i]);
    }
    acc_keys_offset += num_keys;
    acc_vectors_offset += inference_params_.max_batchsize *
                          inference_parser_.max_feature_num_for_tables[i] *
                          inference_parser_.embed_vec_size_for_tables[i];
  }
  for (size_t i = 0; i < num_embedding_tables; ++i) {
    HCTR_LIB_THROW(cudaStreamSynchronize(streams_[i]));
  }

  // convert dense input to dense tensor
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

  acc_vectors_offset = 0;
  acc_row_ptrs_offset = 0;
  for (size_t i = 0; i < num_embedding_tables; ++i) {
    // bind row ptrs input to row ptrs tensor
    auto row_ptrs_dims = row_ptrs_tensors_[i]->get_dimensions();
    std::shared_ptr<TensorBuffer2> row_ptrs_buff =
        PreallocatedBuffer2<int>::create(d_row_ptrs + acc_row_ptrs_offset, row_ptrs_dims);
    bind_tensor_to_buffer(row_ptrs_dims, row_ptrs_buff, row_ptrs_tensors_[i]);
    acc_row_ptrs_offset += num_samples * inference_parser_.slot_num_for_tables[i] + 1;

    // bind embedding vectors from looking up to embedding features tensor
    auto embedding_features_dims = embedding_features_tensors_[i]->get_dimensions();
    std::shared_ptr<TensorBuffer2> embeddding_features_buff = PreallocatedBuffer2<float>::create(
        d_embedding_vectors_ + acc_vectors_offset, embedding_features_dims);
    bind_tensor_to_buffer(embedding_features_dims, embeddding_features_buff,
                          embedding_features_tensors_[i]);
    acc_vectors_offset += inference_params_.max_batchsize *
                          inference_parser_.max_feature_num_for_tables[i] *
                          inference_parser_.embed_vec_size_for_tables[i];
    // feature combiner feedforward
    embedding_feature_combiners_[i]->fprop(false);
  }

  // dense network feedforward
  network_->predict();

  // convert the prediction result to output
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

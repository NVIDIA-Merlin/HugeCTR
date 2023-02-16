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
                                   const std::shared_ptr<EmbeddingCacheBase>& embedding_cache,
                                   std::shared_ptr<ResourceManager> resource_manager)
    : InferenceSessionBase(),
      config_(read_json_file(model_config_path)),
      embedding_table_slot_size_({0}),
      embedding_cache_(embedding_cache),
      inference_parser_(config_),
      inference_params_(inference_params) {
  try {
    if (inference_params_.use_gpu_embedding_cache &&
        embedding_cache->get_device_id() != inference_params_.device_id) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "The device id of inference_params is not consistent with that of embedding cache.");
    }
    resource_manager_ = resource_manager != nullptr
                            ? resource_manager
                            : ResourceManagerCore::create({{inference_params.device_id}}, 0);
    HCTR_LOG(TRACE, ROOT, "Create inference session on device: %d\n", inference_params_.device_id);
    auto b2s = [](const char val) { return val ? "True" : "False"; };
    HCTR_LOG(INFO, ROOT, "Model name: %s\n", inference_params_.model_name.c_str());
    HCTR_LOG(INFO, ROOT, "Use mixed precision: %s\n", b2s(inference_params.use_mixed_precision));
    HCTR_LOG(INFO, ROOT, "Use cuda graph: %s\n", b2s(inference_params.use_cuda_graph));
    HCTR_LOG(INFO, ROOT, "Max batchsize: %lu\n", inference_params.max_batchsize);
    HCTR_LOG(INFO, ROOT, "Use I64 input key: %s\n", b2s(inference_params.i64_input_key));
    Network* network_ptr;
    inference_parser_.create_pipeline(inference_params_, dense_input_tensorbag_, row_ptrs_tensors_,
                                      embedding_features_tensors_, embedding_table_slot_size_,
                                      &embedding_feature_combiners_, &network_ptr,
                                      inference_tensor_entries_, resource_manager_);
    auto dense_network_feedforward =
        std::make_shared<StreamContextScheduleable>([=] { network_->predict(); });
    predict_network_pipeline_ = Pipeline(
        "default", resource_manager_->get_local_gpu_from_device_id(inference_params.device_id),
        {dense_network_feedforward});

    network_ = std::move(std::unique_ptr<Network>(network_ptr));
    network_->initialize(false);
    if (inference_params.use_algorithm_search) {
      network_->search_algorithm();
    }
    if (inference_params_.dense_model_file.size() > 0) {
      network_->upload_params_to_device_inference(inference_params_.dense_model_file);
    }
    if (inference_params_.non_trainable_params_file.size() > 0) {
      network_->upload_non_trainable_params_to_device_inference(
          inference_params_.non_trainable_params_file);
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

    cudaMallocManaged(&d_keys_, inference_params_.max_batchsize *
                                    inference_parser_.max_feature_num_per_sample *
                                    sizeof(long long));
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
  cudaFree(d_keys_);
  for (auto stream : streams_) cudaStreamDestroy(stream);
}

void InferenceSession::predict_impl(float* d_dense, void* keys, bool key_on_device, int* d_row_ptrs,
                                    float* d_output, int num_samples, int num_embedding_tables,
                                    bool table_major_key_layout) {
  CudaDeviceContext context(
      resource_manager_->get_local_gpu_from_device_id(inference_params_.device_id)
          ->get_device_id());
  // embedding_cache lookup
  size_t acc_vectors_offset{0};
  size_t acc_row_ptrs_offset{0};
  size_t acc_keys_offset{0};
  size_t num_keys{0};
  for (size_t i = 0; i < num_embedding_tables; ++i) {
    acc_row_ptrs_offset += num_samples * inference_parser_.slot_num_for_tables[i] + 1;
    num_keys = h_row_ptrs_[acc_row_ptrs_offset - 1];
    if (inference_params_.i64_input_key) {
      if (key_on_device) {
        embedding_cache_->lookup_from_device(i, d_embedding_vectors_ + acc_vectors_offset,
                                             static_cast<const long long*>(keys) + acc_keys_offset,
                                             num_keys, inference_params_.hit_rate_threshold,
                                             streams_[i]);
      } else {
        embedding_cache_->lookup(i, d_embedding_vectors_ + acc_vectors_offset,
                                 static_cast<const long long*>(keys) + acc_keys_offset, num_keys,
                                 inference_params_.hit_rate_threshold, streams_[i]);
      }
    } else {
      if (key_on_device) {
        embedding_cache_->lookup_from_device(
            i, d_embedding_vectors_ + acc_vectors_offset,
            static_cast<const unsigned int*>(keys) + acc_keys_offset, num_keys,
            inference_params_.hit_rate_threshold, streams_[i]);
      } else {
        embedding_cache_->lookup(i, d_embedding_vectors_ + acc_vectors_offset,
                                 static_cast<const unsigned int*>(keys) + acc_keys_offset, num_keys,
                                 inference_params_.hit_rate_threshold, streams_[i]);
      }
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
    convert_array_on_device(
        reinterpret_cast<__half*>(dense_input_tensorbag_.get_ptr()), d_dense, dense_size,
        resource_manager_->get_local_gpu_from_device_id(inference_params_.device_id)->get_stream());

  } else {
    convert_array_on_device(
        reinterpret_cast<float*>(dense_input_tensorbag_.get_ptr()), d_dense, dense_size,
        resource_manager_->get_local_gpu_from_device_id(inference_params_.device_id)->get_stream());
  }

  acc_vectors_offset = 0;
  acc_row_ptrs_offset = 0;
  for (size_t i = 0; i < num_embedding_tables; ++i) {
    // bind row ptrs input to row ptrs tensor
    (*row_ptrs_tensors_[i]) =
        core23::Tensor::bind(d_row_ptrs + acc_row_ptrs_offset, row_ptrs_tensors_[i]->shape(),
                             row_ptrs_tensors_[i]->data_type(), row_ptrs_tensors_[i]->device());
    acc_row_ptrs_offset += num_samples * inference_parser_.slot_num_for_tables[i] + 1;

    // bind embedding vectors from looking up to embedding features tensor
    (*embedding_features_tensors_[i]) = core23::Tensor::bind(
        d_embedding_vectors_ + acc_vectors_offset, embedding_features_tensors_[i]->shape(),
        embedding_features_tensors_[i]->data_type(), embedding_features_tensors_[i]->device());
    acc_vectors_offset += inference_params_.max_batchsize *
                          inference_parser_.max_feature_num_for_tables[i] *
                          inference_parser_.embed_vec_size_for_tables[i];
    // feature combiner feedforward
    embedding_feature_combiners_[i]->fprop(false);
  }

  // dense network feedforward

  if (inference_params_.use_cuda_graph) {
    predict_network_pipeline_.run_graph();
  } else {
    predict_network_pipeline_.run();
  }

  // convert the prediction result to output
  if (inference_params_.use_mixed_precision) {
    convert_array_on_device(
        d_output, network_->get_pred_tensor_half().get_ptr(),
        network_->get_pred_tensor_half().get_num_elements(),
        resource_manager_->get_local_gpu_from_device_id(inference_params_.device_id)->get_stream());
  } else {
    convert_array_on_device(
        d_output, network_->get_pred_tensor().get_ptr(),
        network_->get_pred_tensor().get_num_elements(),
        resource_manager_->get_local_gpu_from_device_id(inference_params_.device_id)->get_stream());
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(
      resource_manager_->get_local_gpu_from_device_id(inference_params_.device_id)->get_stream()));
}

void InferenceSession::predict(float* d_dense, void* h_embeddingcolumns, int* d_row_ptrs,
                               float* d_output, int num_samples, bool table_major_key_layout) {
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
  if (!table_major_key_layout) {
    // HCTR_LOG_S(INFO, ROOT) << "Redistribute keys from sample first to table first" << std::endl;
    if (inference_params_.i64_input_key) {
      distribute_keys_per_table(static_cast<long long*>(h_keys_),
                                static_cast<long long*>(h_embeddingcolumns), h_row_ptrs_,
                                num_samples, inference_parser_.slot_num_for_tables);
    } else {
      distribute_keys_per_table(static_cast<unsigned int*>(h_keys_),
                                static_cast<unsigned int*>(h_embeddingcolumns), h_row_ptrs_,
                                num_samples, inference_parser_.slot_num_for_tables);
    }
  }
  void* h_keys_for_ec = table_major_key_layout ? h_embeddingcolumns : h_keys_;
  predict_impl(d_dense, h_keys_for_ec, false, d_row_ptrs, d_output, num_samples,
               num_embedding_tables, table_major_key_layout);
}

void InferenceSession::predict_from_device(float* d_dense, void* d_embeddingcolumns,
                                           int* d_row_ptrs, float* d_output, int num_samples,
                                           bool table_major_key_layout) {
  size_t num_embedding_tables = inference_parser_.num_embedding_tables;
  if (num_embedding_tables != row_ptrs_tensors_.size() ||
      num_embedding_tables != embedding_features_tensors_.size() ||
      num_embedding_tables != embedding_feature_combiners_.size()) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "embedding feature combiner inconsistent");
  }
  CudaDeviceContext context(
      resource_manager_->get_local_gpu_from_device_id(inference_params_.device_id)
          ->get_device_id());

  cudaStream_t stream =
      resource_manager_->get_local_gpu_from_device_id(inference_params_.device_id)->get_stream();

  // Copy row_ptrs to host
  HCTR_LIB_THROW(cudaMemcpy(
      h_row_ptrs_, d_row_ptrs,
      (num_samples * inference_parser_.slot_num + inference_parser_.num_embedding_tables) *
          sizeof(int),
      cudaMemcpyDeviceToHost));
  // Redistribute keys ：from sample first to table first
  if (!table_major_key_layout) {
    // HCTR_LOG_S(INFO, ROOT) << "Redistribute keys from sample first to table first" << std::endl;
    if (inference_params_.i64_input_key) {
      distribute_keys_per_table_on_device(
          static_cast<long long*>(d_keys_), static_cast<long long*>(d_embeddingcolumns), d_row_ptrs,
          num_samples, inference_parser_.slot_num_for_tables, stream);
    } else {
      distribute_keys_per_table_on_device(
          static_cast<unsigned int*>(d_keys_), static_cast<unsigned int*>(d_embeddingcolumns),
          d_row_ptrs, num_samples, inference_parser_.slot_num_for_tables, stream);
    }
  }

  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  void* d_keys_for_ec = table_major_key_layout ? d_embeddingcolumns : d_keys_;
  predict_impl(d_dense, d_keys_for_ec, true, d_row_ptrs, d_output, num_samples,
               num_embedding_tables, table_major_key_layout);
}

}  // namespace HugeCTR

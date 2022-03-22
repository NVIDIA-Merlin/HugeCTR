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

#include <cpu/create_pipeline_cpu.hpp>
#include <cpu/inference_session_cpu.hpp>
#include <cpu_resource.hpp>
#include <iostream>
#include <vector>
namespace HugeCTR {

template <typename TypeHashKey>
InferenceSessionCPU<TypeHashKey>::InferenceSessionCPU(
    const std::string& model_config_path, const InferenceParams& inference_params,
    const std::shared_ptr<HierParameterServerBase>& parameter_server)
    : config_(read_json_file(model_config_path)),
      embedding_table_slot_size_({0}),
      parameter_server_(parameter_server),
      inference_parser_(config_),
      inference_params_(inference_params) {
  try {
    cpu_resource_.reset(new CPUResource(0, {}));
    NetworkCPU* network_ptr;
    std::map<std::string, bool> tensor_active;

    // create pipeline and initialize network
    create_pipeline_cpu(config_, tensor_active, inference_params_, dense_input_tensor_,
                        row_ptrs_tensors_, embedding_features_tensors_, embedding_table_slot_size_,
                        &embedding_feature_combiners_, &network_ptr, cpu_resource_);
    network_ = std::move(std::unique_ptr<NetworkCPU>(network_ptr));
    network_->initialize();
    if (inference_params_.dense_model_file.size() > 0) {
      network_->load_params_from_model(inference_params_.dense_model_file);
    }

    // allocate memory for embedding vector lookup
    // h_keys_ is a void pointer, which serves key types of both long long and unsigned int
    h_keys_ = malloc(inference_params_.max_batchsize *
                     inference_parser_.max_feature_num_per_sample * sizeof(long long));
    h_embedding_vectors_ =
        (float*)malloc(inference_params_.max_batchsize *
                       inference_parser_.max_embedding_vector_size_per_sample * sizeof(float));
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey>
InferenceSessionCPU<TypeHashKey>::~InferenceSessionCPU() {
  free(h_embedding_vectors_);
  free(h_keys_);
}

template <typename TypeHashKey>
void InferenceSessionCPU<TypeHashKey>::predict(float* h_dense, void* h_embeddingcolumns,
                                               int* h_row_ptrs, float* h_output, int num_samples) {
  size_t num_embedding_tables = inference_parser_.num_embedding_tables;
  if (num_embedding_tables != row_ptrs_tensors_.size() ||
      num_embedding_tables != embedding_features_tensors_.size() ||
      num_embedding_tables != embedding_feature_combiners_.size()) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "embedding feature combiner inconsistent");
  }

  // Redistribute keys ：from sample first to table first
  if (inference_params_.i64_input_key) {
    distribute_keys_per_table(static_cast<long long*>(h_keys_),
                              static_cast<long long*>(h_embeddingcolumns), h_row_ptrs, num_samples,
                              inference_parser_.slot_num_for_tables);
  } else {
    distribute_keys_per_table(static_cast<unsigned int*>(h_keys_),
                              static_cast<unsigned int*>(h_embeddingcolumns), h_row_ptrs,
                              num_samples, inference_parser_.slot_num_for_tables);
  }

  // parameter server lookup
  size_t acc_vectors_offset{0};
  size_t acc_row_ptrs_offset{0};
  size_t acc_keys_offset{0};
  size_t num_keys{0};
  for (size_t i = 0; i < num_embedding_tables; ++i) {
    acc_row_ptrs_offset += num_samples * inference_parser_.slot_num_for_tables[i] + 1;
    num_keys = h_row_ptrs[acc_row_ptrs_offset - 1];
    if (inference_params_.i64_input_key) {
      parameter_server_->lookup(static_cast<const long long*>(h_keys_) + acc_keys_offset, num_keys,
                                h_embedding_vectors_ + acc_vectors_offset,
                                inference_params_.model_name, i);
    } else {
      parameter_server_->lookup(static_cast<const unsigned int*>(h_keys_) + acc_keys_offset,
                                num_keys, h_embedding_vectors_ + acc_vectors_offset,
                                inference_params_.model_name, i);
    }
    acc_keys_offset += num_keys;
    acc_vectors_offset += inference_params_.max_batchsize *
                          inference_parser_.max_feature_num_for_tables[i] *
                          inference_parser_.embed_vec_size_for_tables[i];
  }

  // copy dense input to dense tensor
  auto dense_dims = dense_input_tensor_.get_dimensions();
  size_t dense_size = 1;
  for (auto dim : dense_dims) {
    dense_size *= dim;
  }
  size_t dense_size_in_bytes = dense_size * sizeof(float);
  memcpy(dense_input_tensor_.get_ptr(), h_dense, dense_size_in_bytes);

  acc_vectors_offset = 0;
  acc_row_ptrs_offset = 0;
  for (size_t i = 0; i < num_embedding_tables; ++i) {
    // bind row ptrs input to row ptrs tensor
    auto row_ptrs_dims = row_ptrs_tensors_[i]->get_dimensions();
    std::shared_ptr<TensorBuffer2> row_ptrs_buff =
        PreallocatedBuffer2<int>::create(h_row_ptrs + acc_row_ptrs_offset, row_ptrs_dims);
    bind_tensor_to_buffer(row_ptrs_dims, row_ptrs_buff, row_ptrs_tensors_[i]);
    acc_row_ptrs_offset += num_samples * inference_parser_.slot_num_for_tables[i] + 1;

    // bind embedding vectors from looking up to embedding features tensor
    auto embedding_features_dims = embedding_features_tensors_[i]->get_dimensions();
    std::shared_ptr<TensorBuffer2> embeddding_features_buff = PreallocatedBuffer2<float>::create(
        h_embedding_vectors_ + acc_vectors_offset, embedding_features_dims);
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

  // copy the prediction result to output
  float* h_pred = network_->get_pred_tensor().get_ptr();
  memcpy(h_output, h_pred, network_->get_pred_tensor().get_num_elements() * sizeof(float));
}

template class InferenceSessionCPU<unsigned int>;
template class InferenceSessionCPU<long long>;

}  // namespace HugeCTR

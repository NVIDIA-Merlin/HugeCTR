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
#include <cpu/session_inference_cpu.hpp>
#include <cpu_resource.hpp>
#include <iostream>
#include <vector>
namespace HugeCTR {

template <typename TypeHashKey>
InferenceSessionCPU<TypeHashKey>::InferenceSessionCPU(
    const std::string& config_file, const InferenceParams& inference_params,
    std::shared_ptr<HugectrUtility<TypeHashKey>>& ps)
    : config_(read_json_file(config_file)),
      embedding_table_slot_size_({0}),
      parameter_server_(ps),
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
    h_embeddingvectors_ =
        (float*)malloc(inference_params_.max_batchsize *
                       inference_parser_.max_embedding_vector_size_per_sample * sizeof(float));
    if (inference_params_.i64_input_key) {
      h_shuffled_embeddingcolumns_ =
          malloc(inference_params_.max_batchsize * inference_parser_.max_feature_num_per_sample *
                 sizeof(long long));
    } else {
      h_shuffled_embeddingcolumns_ =
          malloc(inference_params_.max_batchsize * inference_parser_.max_feature_num_per_sample *
                 sizeof(unsigned int));
    }
    h_shuffled_embedding_offset_ =
        (size_t*)malloc((inference_parser_.num_embedding_tables + 1) * sizeof(size_t));
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
  return;
}  // namespace HugeCTR

template <typename TypeHashKey>
InferenceSessionCPU<TypeHashKey>::~InferenceSessionCPU() {
  free(h_embeddingvectors_);
  free(h_shuffled_embeddingcolumns_);
  free(h_shuffled_embedding_offset_);
}

template <typename TypeHashKey>
void InferenceSessionCPU<TypeHashKey>::separate_keys_by_table_(
    int* h_row_ptrs, const std::vector<size_t>& embedding_table_slot_size, int num_samples) {
  size_t slot_num = inference_parser_.slot_num;
  size_t num_embedding_tables = inference_parser_.num_embedding_tables;
  h_embedding_offset_.resize(num_samples * num_embedding_tables + 1);
  for (int i = 0; i < num_samples; i++) {
    for (int j = 0; j < static_cast<int>(num_embedding_tables); j++) {
      h_embedding_offset_[i * num_embedding_tables + j + 1] =
          h_row_ptrs[i * slot_num + static_cast<int>(embedding_table_slot_size[j + 1])];
    }
  }
}

template <typename TypeHashKey>
void InferenceSessionCPU<TypeHashKey>::look_up_(const void* h_embeddingcolumns,
                                                const std::vector<size_t>& h_embedding_offset,
                                                float* h_embeddingvectors) {
  // Shuffle the input embeddingcolumns
  size_t num_sample = (h_embedding_offset.size() - 1) / inference_parser_.num_embedding_tables;
  size_t acc_offset = 0;
  for (unsigned int i = 0; i < inference_parser_.num_embedding_tables; i++) {
    h_shuffled_embedding_offset_[i] = acc_offset;
    for (unsigned int j = 0; j < num_sample; j++) {
      TypeHashKey* dst_ptr = (TypeHashKey*)(h_shuffled_embeddingcolumns_) + acc_offset;
      TypeHashKey* src_prt = (TypeHashKey*)(h_embeddingcolumns) +
                             h_embedding_offset[j * inference_parser_.num_embedding_tables + i];
      size_t cpy_len = h_embedding_offset[j * inference_parser_.num_embedding_tables + i + 1] -
                       h_embedding_offset[j * inference_parser_.num_embedding_tables + i];
      size_t cpy_len_in_byte = cpy_len * sizeof(TypeHashKey);
      memcpy(dst_ptr, src_prt, cpy_len_in_byte);
      acc_offset += cpy_len;
    }
  }
  h_shuffled_embedding_offset_[inference_parser_.num_embedding_tables] = acc_offset;
  if (h_shuffled_embedding_offset_[inference_parser_.num_embedding_tables] !=
      h_embedding_offset[num_sample * inference_parser_.num_embedding_tables]) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Error: embeddingcolumns buffer size is not consist before and after shuffle.");
  }

  // look up
  size_t acc_emb_vec_offset = 0;
  for (unsigned int i = 0; i < inference_parser_.num_embedding_tables; i++) {
    TypeHashKey* h_query_key_ptr =
        (TypeHashKey*)(h_shuffled_embeddingcolumns_) + h_shuffled_embedding_offset_[i];
    size_t query_length = h_shuffled_embedding_offset_[i + 1] - h_shuffled_embedding_offset_[i];
    size_t query_length_in_float = query_length * inference_parser_.embed_vec_size_for_tables[i];
    float* h_vals_retrieved_ptr = h_embeddingvectors + acc_emb_vec_offset;
    parameter_server_->look_up(h_query_key_ptr, query_length, h_vals_retrieved_ptr,
                               inference_params_.model_name, i);
    acc_emb_vec_offset += query_length_in_float;
  }
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

  // embedding cache look up and update
  separate_keys_by_table_(h_row_ptrs, embedding_table_slot_size_, num_samples);
  look_up_(h_embeddingcolumns, h_embedding_offset_, h_embeddingvectors_);

  // copy dense input to dense tensor
  auto dense_dims = dense_input_tensor_.get_dimensions();
  size_t dense_size = 1;
  for (auto dim : dense_dims) {
    dense_size *= dim;
  }
  size_t dense_size_in_bytes = dense_size * sizeof(float);
  memcpy(dense_input_tensor_.get_ptr(), h_dense, dense_size_in_bytes);

  // bind row ptrs input to row ptrs tensor
  auto row_ptrs_dims = row_ptrs_tensors_[0]->get_dimensions();
  std::shared_ptr<TensorBuffer2> row_ptrs_buff =
      PreallocatedBuffer2<int>::create(h_row_ptrs, row_ptrs_dims);
  bind_tensor_to_buffer(row_ptrs_dims, row_ptrs_buff, row_ptrs_tensors_[0]);

  // bind embedding vectors from looking up to embedding features tensor
  auto embedding_features_dims = embedding_features_tensors_[0]->get_dimensions();
  std::shared_ptr<TensorBuffer2> embeddding_features_buff =
      PreallocatedBuffer2<float>::create(h_embeddingvectors_, embedding_features_dims);
  bind_tensor_to_buffer(embedding_features_dims, embeddding_features_buff,
                        embedding_features_tensors_[0]);

  // feature combiner & dense network feedforward, they are both using
  // resource_manager_->get_local_gpu(0)->get_stream()
  embedding_feature_combiners_[0]->fprop(false);
  network_->predict();

  // copy the prediction result to output
  float* h_pred = network_->get_pred_tensor().get_ptr();
  memcpy(h_output, h_pred, inference_params_.max_batchsize * sizeof(float));
}

template class InferenceSessionCPU<unsigned int>;
template class InferenceSessionCPU<long long>;

}  // namespace HugeCTR

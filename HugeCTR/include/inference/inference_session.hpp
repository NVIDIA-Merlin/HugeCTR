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

#pragma once
#include <common.hpp>
#include <hps/embedding_cache_base.hpp>
#include <inference/inference_session_base.hpp>
#include <inference/preallocated_buffer2.hpp>
#include <metrics.hpp>
#include <network.hpp>
#include <parser.hpp>
#include <string>
#include <tensor2.hpp>
#include <thread>
#include <utility>
#include <vector>

#include "pipeline.hpp"

namespace HugeCTR {

class InferenceSession : public InferenceSessionBase {
 private:
  nlohmann::json config_;  // should be declared before parser_ and inference_parser_
  std::vector<size_t> embedding_table_slot_size_;
  std::vector<cudaStream_t> streams_;

  std::vector<std::shared_ptr<Tensor2<int>>> row_ptrs_tensors_;  // embedding input row
  std::vector<std::shared_ptr<Tensor2<float>>>
      embedding_features_tensors_;                     // embedding input value vector
  TensorBag2 dense_input_tensorbag_;                   // dense input vector
  std::vector<TensorEntry> inference_tensor_entries_;  // tensor entries in the inference pipeline

  std::vector<std::shared_ptr<Layer>> embedding_feature_combiners_;
  std::unique_ptr<Network> network_;
  std::shared_ptr<EmbeddingCacheBase> embedding_cache_;

  int* h_row_ptrs_;
  void* h_keys_;

  int* d_row_ptrs_;
  void* d_keys_;
  float* d_embedding_vectors_;

  Pipeline predict_network_pipeline_;

  void predict_impl(float* d_dense, void* keys, bool key_on_device, int* d_row_ptrs,
                    float* d_output, int num_samples, int num_embedding_tables,
                    bool table_major_key_layout);

 protected:
  InferenceParser inference_parser_;
  InferenceParams inference_params_;
  std::shared_ptr<IDataReader> data_reader_;
  std::shared_ptr<ResourceManager> resource_manager_;

 public:
  virtual ~InferenceSession();
  InferenceSession(const std::string& model_config_path, const InferenceParams& inference_params,
                   const std::shared_ptr<EmbeddingCacheBase>& embedding_cache,
                   std::shared_ptr<ResourceManager> resource_manager = nullptr);
  InferenceSession(InferenceSession const&) = delete;
  InferenceSession& operator=(InferenceSession const&) = delete;

  virtual void predict(float* d_dense, void* h_embeddingcolumns, int* d_row_ptrs, float* d_output,
                       int num_samples, bool table_major_key_layout = false);

  virtual void predict_from_device(float* d_dense, void* d_embeddingcolumns, int* d_row_ptrs,
                                   float* d_output, int num_samples,
                                   bool table_major_key_layout = false);

  const InferenceParser& get_inference_parser() const { return inference_parser_; }
  const std::vector<TensorEntry>& get_inference_tensor_entries() const {
    return inference_tensor_entries_;
  }
};

}  // namespace HugeCTR

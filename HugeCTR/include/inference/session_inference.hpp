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

#pragma once
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/inference/hugectrmodel.hpp"
#include "HugeCTR/include/inference/preallocated_buffer2.hpp"
#include "HugeCTR/include/metrics.hpp"
#include "HugeCTR/include/network.hpp"
#include "HugeCTR/include/parser.hpp"
#include "HugeCTR/include/tensor2.hpp"
namespace HugeCTR {

class InferenceSession : public HugeCTRModel {
  nlohmann::json config_; // should be declared before parser_ and inference_parser_
  Parser parser_;
  InferenceParser inference_parser_;
  std::vector<size_t> embedding_table_slot_size_;
  std::vector<cudaStream_t> lookup_streams_;
  std::vector<cudaStream_t> update_streams_;

  std::vector<std::shared_ptr<Tensor2<int>>> row_ptrs_tensors_; // embedding input row
  std::vector<std::shared_ptr<Tensor2<float>>> embedding_features_tensors_; // embedding input value vector
  Tensor2<float> dense_input_tensor_;  // dense input vector

  std::vector<std::shared_ptr<Layer>> embedding_feature_combiners_;
  std::unique_ptr<Network> network_;
  std::shared_ptr<ResourceManager> resource_manager_;
  embedding_interface* embedding_cache_;

  std::vector<size_t> h_embedding_offset_; // embedding offset to indicate which embeddingcolumns belong to the same embedding table
  std::vector<int*> d_row_ptrs_vec_; // row ptrs (on device) for each embedding table

  embedding_cache_workspace workspace_handler_;
  float* d_embeddingvectors_;
  
  void separate_keys_by_table_(int* d_row_ptrs, const std::vector<size_t>& embedding_table_slot_size, int num_samples);

 public:
  InferenceSession(const std::string& config_file, int device_id, embedding_interface* embedding_ptr);
  virtual ~InferenceSession();
  void predict(float* d_dense, void* h_embeddingcolumns, int* d_row_ptrs, float* d_output, int num_samples);
};

}  // namespace HugeCTR
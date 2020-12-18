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
#include "HugeCTR/include/embedding.hpp"
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

  Tensors2<int> row_ptrs_tensors_; // embedding input row
  Tensors2<float> embedding_features_tensors_; // embedding input value vector

  std::vector<std::shared_ptr<Layer>> embedding_feature_combiners_;
  std::unique_ptr<Network> network_;
  std::shared_ptr<ResourceManager> resource_manager_;

 public:
  InferenceSession(const std::string& config_file, cudaStream_t stream);
  virtual ~InferenceSession();
  void predict(float* d_dense, int* d_row_ptrs, float* d_embeddingvectors, float* d_output, int num_samples);
};

}  // namespace HugeCTR

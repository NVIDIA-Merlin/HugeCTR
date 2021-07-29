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
#include <cpu/embedding_feature_combiner_cpu.hpp>
#include <cpu/network_cpu.hpp>
#include <inference/hugectrmodel.hpp>
#include <inference/inference_utils.hpp>
#include <inference/preallocated_buffer2.hpp>
#include <parser.hpp>
#include <string>
#include <tensor2.hpp>
#include <thread>
#include <utility>
#include <vector>

namespace HugeCTR {

template <typename TypeHashKey>
class InferenceSessionCPU : public HugeCTRModel {
 private:
  nlohmann::json config_;
  std::string model_name_;
  std::vector<size_t> embedding_table_slot_size_;

  std::vector<std::shared_ptr<Tensor2<int>>> row_ptrs_tensors_;
  std::vector<std::shared_ptr<Tensor2<float>>> embedding_features_tensors_;
  Tensor2<float> dense_input_tensor_;

  std::vector<std::shared_ptr<LayerCPU>> embedding_feature_combiners_;
  std::unique_ptr<NetworkCPU> network_;
  std::shared_ptr<HugectrUtility<TypeHashKey>> parameter_server_;

  std::vector<size_t> h_embedding_offset_;
  std::vector<int*> h_row_ptrs_vec_;

  float* h_embeddingvectors_;
  void* h_shuffled_embeddingcolumns_;
  size_t* h_shuffled_embedding_offset_;

  std::shared_ptr<CPUResource> cpu_resource_;

  void separate_keys_by_table_(int* h_row_ptrs,
                               const std::vector<size_t>& embedding_table_slot_size,
                               int num_samples);
  void look_up_(const void* h_embeddingcolumns, const std::vector<size_t>& h_embedding_offset,
                float* h_embeddingvectors);

 protected:
  InferenceParser inference_parser_;
  InferenceParams inference_params_;

 public:
  InferenceSessionCPU(const std::string& model_config_path, const InferenceParams& inference_params,
                      std::shared_ptr<HugectrUtility<TypeHashKey>>& ps);
  virtual ~InferenceSessionCPU();
  void predict(float* h_dense, void* h_embeddingcolumns, int* h_row_ptrs, float* h_output,
               int num_samples);
};

}  // namespace HugeCTR
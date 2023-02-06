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
#pragma once

#include <common.hpp>
#include <cpu/embedding_feature_combiner_cpu.hpp>
#include <cpu/network_cpu.hpp>
#include <hps/hier_parameter_server.hpp>
#include <inference/preallocated_buffer2.hpp>
#include <parser.hpp>
#include <string>
#include <tensor2.hpp>
#include <thread>
#include <utility>
#include <vector>

namespace HugeCTR {

template <typename TypeHashKey>
class InferenceSessionCPU {
 private:
  nlohmann::json config_;
  std::string model_name_;
  std::vector<size_t> embedding_table_slot_size_;

  std::vector<std::shared_ptr<Tensor2<int>>> row_ptrs_tensors_;
  std::vector<std::shared_ptr<Tensor2<float>>> embedding_features_tensors_;
  Tensor2<float> dense_input_tensor_;

  std::vector<std::shared_ptr<LayerCPU>> embedding_feature_combiners_;
  std::unique_ptr<NetworkCPU> network_;
  std::shared_ptr<HierParameterServerBase> parameter_server_;

  void* h_keys_;
  float* h_embedding_vectors_;

  std::shared_ptr<CPUResource> cpu_resource_;

 protected:
  InferenceParser inference_parser_;
  InferenceParams inference_params_;

 public:
  InferenceSessionCPU(const std::string& model_config_path, const InferenceParams& inference_params,
                      const std::shared_ptr<HierParameterServerBase>& parameter_server);
  virtual ~InferenceSessionCPU();
  void predict(float* h_dense, void* h_embeddingcolumns, int* h_row_ptrs, float* h_output,
               int num_samples);
};

}  // namespace HugeCTR
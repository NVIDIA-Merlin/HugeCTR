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
#include <parser.hpp>
#include <cpu/layer_cpu.hpp>
#include <cpu/network_cpu.hpp>

namespace HugeCTR {

void create_pipeline_cpu(const nlohmann::json& config,
                      std::map<std::string, bool> tensor_active,
                      const InferenceParams& inference_params,
                      Tensor2<float>& dense_input,
                      std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                      std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                      std::vector<size_t>& embedding_table_slot_size,
                      std::vector<std::shared_ptr<LayerCPU>>* embeddings, NetworkCPU** network,
                      const std::shared_ptr<CPUResource>& cpu_resource);

template <typename TypeEmbeddingComp>
void create_pipeline_inference_cpu(const nlohmann::json& config,
                                  const InferenceParams& inference_params,
                                  Tensor2<float>& dense_input,
                                  std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                                  std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                                  std::vector<size_t>& embedding_table_slot_size,
                                  std::vector<std::shared_ptr<LayerCPU>>* embeddings,
                                  NetworkCPU** network,
                                  const std::shared_ptr<CPUResource>& cpu_resource);

} // namespace HugeCTR
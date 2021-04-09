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
#include <cpu/embedding_feature_combiner_cpu.hpp>
#include <inference/preallocated_buffer2.hpp>

namespace HugeCTR {

template <typename TypeFP>
struct create_embedding_cpu {
  void operator()(const InferenceParser& inference_parser, const nlohmann::json& j_layers_array,
                  std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                  std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                  std::vector<size_t>& embedding_table_slot_size,
                  std::vector<TensorEntry>* tensor_entries,
                  std::vector<std::shared_ptr<LayerCPU>>* embeddings,
                  std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff);
};

} // namespace HugeCTR
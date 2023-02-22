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

#include <core23/tensor.hpp>
#include <metrics.hpp>
#include <optional>
#include <vector>

namespace HugeCTR {

class ILoss;
struct DenseLayer;
class Layer;

struct TensorEntity {
  std::string name;
  core23::Tensor tensor;
};

std::optional<core23::Tensor> get_tensor_from_entities(
    const std::vector<TensorEntity> tensor_entities, const std::string& name);

struct InputTensorsAndOutputNames {
  std::vector<core23::Tensor> input_tensors;
  std::vector<std::string> output_names;
};

InputTensorsAndOutputNames get_input_tensors_and_output_names(
    const std::vector<std::string>& bottom_names, const std::vector<std::string>& top_names,
    const std::vector<TensorEntity>& tensor_entities);

void add_dense_layer_impl(DenseLayer& dense_layer, std::vector<TensorEntity>& tensor_entities,
                          std::vector<std::unique_ptr<Layer>>& layers,
                          std::map<std::string, core23::Tensor>& loss_tensors,
                          std::map<std::string, std::unique_ptr<ILoss>>& losses,
                          bool async_mlp_wgrad, metrics::Core23MultiLossMetricMap* raw_metrics,
                          int gpu_count_in_total, const std::shared_ptr<GPUResource>& gpu_resource,
                          bool use_mixed_precision, bool enable_tf32_compute, float scaler,
                          bool use_algorithm_search,
                          std::vector<Layer*>* embedding_dependent_layers,
                          std::vector<Layer*>* embedding_independent_layers,
                          bool embedding_dependent);
}  // namespace HugeCTR

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

#include <core23/logger.hpp>
#include <core23/tensor.hpp>
#include <layer.hpp>
#include <pybind/dense_layer_helpers.hpp>
#include <pybind/model.hpp>
#include <string>
#include <utest/test_utils.hpp>
#include <vector>

namespace HugeCTR::test {

template <typename LayerType>
void dense_layer_test_common(std::vector<TensorEntity> tensor_entities, DenseLayer &dense_layer,
                             bool use_mixed_precision, bool enable_tf32_compute, bool async_wgrad,
                             float scaler, bool use_algorithm_search, bool embedding_dependent) {
  int gpu_count_in_total = 1;

  std::vector<std::unique_ptr<Layer>> layers;
  std::map<std::string, core23::Tensor> loss_tensors;
  std::map<std::string, std::unique_ptr<ILoss>> losses;

  metrics::Core23MultiLossMetricMap raw_metrics;

  std::vector<Layer *> embedding_dependent_layers;
  std::vector<Layer *> embedding_independent_layers;

  int64_t num_tensors = dense_layer.bottom_names.size() + dense_layer.top_names.size();

  dense_layer.compute_config.async_wgrad = async_wgrad;

  add_dense_layer_impl(dense_layer, tensor_entities, layers, loss_tensors, losses, &raw_metrics,
                       gpu_count_in_total, test::get_default_gpu(), use_mixed_precision,
                       enable_tf32_compute, scaler, use_algorithm_search,
                       &embedding_dependent_layers, &embedding_independent_layers,
                       embedding_dependent);

  EXPECT_TRUE(tensor_entities.size() == num_tensors);
  EXPECT_TRUE(layers.size() == 1);
  EXPECT_TRUE(dynamic_cast<LayerType *>(layers[0].get()) != nullptr);
}

}  // namespace HugeCTR::test
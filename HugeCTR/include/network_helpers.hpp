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

#include <core23/tensor_container.hpp>
#include <trainable_layer.hpp>

namespace HugeCTR {

using WeightTensors = core23::TensorContainer<float, 1, 1>;
using WeightHalfTensors = core23::TensorContainer<__half, 1, 1>;
template <typename T>
using WgradTensors = core23::TensorContainer<T, 1, 1>;

template <typename DType, typename Range>
std::vector<core23::Tensor> get_trainable_tensor_vector(
    const std::vector<std::unique_ptr<Layer>>& layers, Range range) {
  std::vector<core23::Tensor> param_tensors;
  auto op = [&param_tensors, range](auto trainable_layer) {
    if (trainable_layer) {
      for (auto& param_tensor : range(trainable_layer)) {
        param_tensors.push_back(param_tensor);
      }
      return true;
    } else {
      return false;
    }
  };
  for (auto& layer : layers) {
    auto trainable_layer = dynamic_cast<Core23TempTrainableLayer<DType>*>(layer.get());
    if (!op(trainable_layer)) {
      auto trainable_layer = dynamic_cast<Core23TempTrainableLayer<DType, true>*>(layer.get());
      op(trainable_layer);
    }
  }
  return param_tensors;
}

template <typename DType>
std::vector<core23::Tensor> get_weight_tensor_vector(
    const std::vector<std::unique_ptr<Layer>>& layers) {
  return get_trainable_tensor_vector<DType>(
      layers, [](auto& layer) -> auto { return layer->get_weights(); });
}

template <typename DType>
std::vector<core23::Tensor> get_master_weight_tensor_vector(
    const std::vector<std::unique_ptr<Layer>>& layers) {
  return get_trainable_tensor_vector<DType>(
      layers, [](auto& layer) -> auto { return layer->get_master_weights(); });
}

template <typename DType>
std::vector<core23::Tensor> get_wgrad_tensor_vector(
    const std::vector<std::unique_ptr<Layer>>& layers) {
  return get_trainable_tensor_vector<DType>(
      layers, [](auto& layer) -> auto { return layer->get_wgrads(); });
}

template <typename DType, typename WType, typename Range>
std::optional<core23::TensorContainer<WType, 1, 1>> get_trainable_tensors(
    const std::vector<std::unique_ptr<Layer>>& layers, Range range) {
  std::vector<core23::Tensor> param_tensors = get_trainable_tensor_vector<DType>(layers, range);
  return std::make_optional<core23::TensorContainer<WType, 1, 1>>(
      std::move(param_tensors), core23::Shape({static_cast<int64_t>(param_tensors.size())}));
}

}  // namespace HugeCTR

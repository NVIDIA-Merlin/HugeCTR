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

#include <layer.hpp>
#include <vector>

namespace HugeCTR {
namespace reshape_layer_utils {

std::vector<int64_t> calc_output_shape(const std::vector<int64_t>& input_shape,
                                       const std::vector<int64_t>& out_shape_with_placeholder);
}  // namespace reshape_layer_utils

/**
 * Layer which reshapes a 3D/2D input tensor to 2D output tensor,
 * e.g., (batch_size, n_slots, vector_size) to (batch_size, n_slots * vector_size),
 * e.g., (batch_size * n_slots, vector_size) to (batch_size, n_slots * vector_size),
 * If the input tensor is 3D, you can choose which slots participate by calling the different Ctor
 */
template <typename T>
class ReshapeLayerV2 : public Layer {
 public:
  ReshapeLayerV2(const core23::Tensor& input_tensor, core23::Tensor& output_tensor,
                 const std::vector<int64_t>& out_shape_with_placeholder,
                 const std::shared_ptr<GPUResource>& gpu_resource);

  /**
   * A method of implementing the forward pass of Reshape
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * A method of implementing the forward pass of Reshape
   * @param stream CUDA stream where the forward propagation is executed
   */
  void bprop() override;
};

}  // namespace HugeCTR

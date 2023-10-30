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
#include <set>
#include <vector>

namespace HugeCTR {

template <typename T>
class GatherLayer : public Layer {
 public:
  GatherLayer(const core23::Tensor& input_tensor, core23::Tensor& output_tensor,
              std::vector<int>& indices, const std::shared_ptr<GPUResource>& gpu_resource);
  void initialize() override;
  /**
   * Gather's forward pass to gather data to the output tensor
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * Gather's backward pass to scatter data to the input tensors
   * @param stream CUDA stream where the forward propagation is executed
   */
  void bprop() override;

 private:
  size_t tensor_size_;
  size_t num_indices_;
  std::vector<int> h_indices_;
  core23::Tensor indices23_;
};

}  // namespace HugeCTR

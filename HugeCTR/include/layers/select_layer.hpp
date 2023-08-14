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
/**
 * Layer which reshapes a 3D/2D input tensor to 2D output tensor,
 * e.g., (batch_size, n_slots, vector_size) to (batch_size, n_slots * vector_size),
 * e.g., (batch_size * n_slots, vector_size) to (batch_size, n_slots * vector_size),
 * If the input tensor is 3D, you can choose which slots participate by calling the different Ctor
 */
template <typename T>
class SelectLayer : public Layer {
  int64_t num_before_selected_dim_;
  int64_t num_in_selected_dim_;
  int64_t num_after_selected_dim_;
  std::vector<int64_t> h_index_;
  core23::Tensor d_index_;

  void prop_common(bool forward, bool is_train, cudaStream_t stream);

 public:
  SelectLayer(const core23::Tensor& input_tensor, core23::Tensor& output_tensor, int dim,
              const std::vector<int64_t>& index, const std::shared_ptr<GPUResource>& gpu_resource);

  void initialize() override;

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

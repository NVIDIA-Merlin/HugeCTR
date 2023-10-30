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
 * Layer which
 */
template <typename T>
class InteractionLayer : public Layer {
  bool enable_tf32_compute_;

  bool use_mixed_precision_;

  bool separate_Y_and_dY_;

  std::vector<core23::Tensor> intermediate_tensors_;

 private:
  void init(const core23::Tensor& input_bottom_mlp_tensor, const core23::Tensor& input_embeddings,
            core23::Tensor& output_tensor, core23::Tensor& grad_tensor,
            const std::shared_ptr<GPUResource>& gpu_resource);

 public:
  InteractionLayer(const core23::Tensor& input_bottom_mlp_tensor,
                   const core23::Tensor& input_embeddings, core23::Tensor& out_tensor,
                   const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision,
                   bool enable_tf32_compute);

  InteractionLayer(const core23::Tensor& input_bottom_mlp_tensor,
                   const core23::Tensor& input_embeddings, core23::Tensor& out_tensor,
                   core23::Tensor& grad_tensor, const std::shared_ptr<GPUResource>& gpu_resource,
                   bool use_mixed_precision, bool enable_tf32_compute);

  ~InteractionLayer() override;

  /**
   * Interaction's forward pass to gather data to the output tensor
   * @param stream CUDA stream where the forward propagation is executed
   */

  void fprop_generic(bool is_train);
  void fprop(bool is_train) override;
  /**
   * Interaction's backward pass to scatter data to the input tensors
   * @param stream CUDA stream where the forward propagation is executed
   */
  void bprop_generic();
  void bprop() override;
  core23::Tensor& get_intermediate(int64_t i) { return intermediate_tensors_[i]; };
};

}  // namespace HugeCTR

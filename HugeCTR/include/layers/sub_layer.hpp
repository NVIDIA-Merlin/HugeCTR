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
#include <layer.hpp>
#include <vector>

namespace HugeCTR {

/**
 * Layer which does element-wise Sub by input tensors.
 * All the input tensors should have the same shape.
 */
template <typename T>
class SubLayer : public Layer {
 public:
  SubLayer(const std::vector<core23::Tensor>& input_tensors, const core23::Tensor& output_tensor,
           const std::shared_ptr<GPUResource>& gpu_resource);
  void initialize() override;

  /**
   * SubLayer's forward propagation
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * SubLayer's backward propagation
   * @param stream CUDA stream where the forward propagation is executed
   */
  void bprop() override;

 private:
  int size_;
  size_t num_;
  core23::Tensor input_tensor_ptr_;
};

}  // namespace HugeCTR

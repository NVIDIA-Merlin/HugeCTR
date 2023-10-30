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
 * Layer which does element-wise Sub by input tensors.
 * All the input tensors should have the same shape.
 */
template <typename T>
class MatrixMultiplyLayer : public Layer {
 public:
  MatrixMultiplyLayer(const std::vector<core23::Tensor>& input_tensors,
                      core23::Tensor& output_tensor,
                      const std::shared_ptr<GPUResource>& gpu_resource);
  /**
   * MatrixMultiplyLayer's forward propagation
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * MatrixMultiplyLayer's backward propagation
   * @param stream CUDA stream where the forward propagation is executed
   */
  void bprop() override;

 private:
  size_t num_;
  size_t dims_lhs_;
  size_t dims_rhs_;
  core23::Tensor fprop_inputA_tensor23_;
};

}  // namespace HugeCTR

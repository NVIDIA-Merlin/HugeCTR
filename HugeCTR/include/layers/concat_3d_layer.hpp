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
#include <optional>
#include <vector>

namespace HugeCTR {

/**
 * Layer which merges the multiple 3D input tensors to a single 3D output tensor along dimension
 * axis. The input tensors and the resulting output tensor must have the same dimensionallity. Only
 * the innermost dimension is expanded by concatenating those of the input tensors. e.g., along axis
 * 1, 3X(batch_size, n_slots, vector_length) to (batch_size, 3 * n_slots , vector_length), e.g.,
 * along axis 2, (batch_size, n_slots, vector_length1) + (batch_size, n_slots, vector_length2) to
 * (batch_size, n_slots, vector_length1 + vector_lenght2)
 */
template <typename T>
class Concat3DLayer : public Layer {
 public:
  /**
   * Ctor of ConcatLayer.
   * @param in_tensors the vector of the input tensors
   * @param out_tensor the resulting output tensor
   * @param blobs_buff GeneralBuffer used to create the output tensor
   * @param axis Dimension along which to concatenate. Must be 1 or 2
   * @param device_id the id of GPU where this layer belongs
   */
  Concat3DLayer(const Tensors2<T>& in_tensors, Tensor2<T>& out_tensor,
                const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff, int axis,
                const std::shared_ptr<GPUResource>& gpu_resource);
  Concat3DLayer(std::vector<core23::Tensor>& in_tensors, core23::Tensor& out_tensor, int axis,
                const std::shared_ptr<GPUResource>& gpu_resource);
  ~Concat3DLayer() override{};

  void initialize() override;

  /**
   * Concat's foward pass to gather data to the output tensor
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * Concat's backward pass to scatter data to the input tensors
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop() override;

 private:
  /*
   * stores the references to the input tensors of this layer.
   */
  Tensors2<T> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensor2<T> out_tensor_;
  /*
   * stores the axis.
   */
  int axis_;
  std::vector<size_t> h_vecs_size_;
  size_t new_width_ = 0;
  size_t num_;
  size_t batch_size_ = 0;
  size_t new_slot_num_ = 0;
  Tensor2<size_t> vecs_size_;
  Tensor2<T*> h_inputs_;
  Tensor2<T*> d_inputs_;
  core23::TensorContainer<T, 1, 1> input_tensor_container_;
};

}  // namespace HugeCTR

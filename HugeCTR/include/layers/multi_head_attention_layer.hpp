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
 * Layer which does mult-head attention by input tensors.
 * All the input tensors should have the same shape.
 */
template <typename T>
class MultiHeadAttentionLayer : public Layer {
 public:
  MultiHeadAttentionLayer(const std::vector<core23::Tensor>& input_tensors,
                          std::vector<core23::Tensor>& output_tensors, int num_attention_heads,
                          bool transpose_b, const std::shared_ptr<GPUResource>& gpu_resource,
                          bool use_mixed_precision, bool enable_tf32_compute);
  /**
   * Ctor of MultiHeadAttentionLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the resulting output tensor
   * @param blobs_buff GeneralBuffer used to create the output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  MultiHeadAttentionLayer(const Tensors2<T>& in_tensors, Tensors2<T>& out_tensor,
                          const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                          int num_attention_heads, bool transpose_b,
                          const std::shared_ptr<GPUResource>& gpu_resource,
                          bool use_mixed_precision, bool enable_tf32_compute);

  // void initialize() override;
  /**
   * MultiHeadAttentionLayer's foward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * MultiHeadAttentionLayer's backward propagation
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
  Tensors2<T> out_tensors_;
  /*
   * stores the axis.
   */

  bool enable_tf32_compute_;
  bool use_mixed_precision_;
  int64_t num_;
  int64_t dims_;
  bool transpose_b_;
  int64_t num_head_;
  Tensor2<T> fprop_inputA_;
  Tensor2<T> query_buf_;
  Tensor2<T> key_buf_;
  Tensor2<T> value_buf_;
  core23::Tensor fprop_inputA_tensor_;
  core23::Tensor query_buf_tensor_;
  core23::Tensor key_buf_tensor_;
  core23::Tensor value_buf_tensor_;
};

}  // namespace HugeCTR

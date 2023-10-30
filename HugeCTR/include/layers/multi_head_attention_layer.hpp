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
#include <layers/masked_softmax_layer.hpp>
#include <layers/softmax_layer.hpp>
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
  // void initialize() override;
  /**
   * MultiHeadAttentionLayer's forward propagation
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * MultiHeadAttentionLayer's backward propagation
   * @param stream CUDA stream where the forward propagation is executed
   */
  void bprop() override;

  std::vector<T>& get_debug_vector() { return debug_vector_; };

 private:
  bool enable_tf32_compute_;
  bool use_mixed_precision_;
  int64_t num_;
  int64_t dims_;
  bool transpose_b_;
  int64_t num_head_;

  core23::Tensor fprop_query_tensor_;
  core23::Tensor fprop_softmax_tensor_;
  core23::Tensor query_buf_tensor_;
  core23::Tensor key_buf_tensor_;
  core23::Tensor attention_out_4d_;

  core23::Tensor attention_value_4d_;
  core23::Tensor attention_score_4d_;
  core23::Tensor attention_softmax_4d_;

  // masked_softmax_layer_ xor softmax_layer_
  std::unique_ptr<MaskedSoftmaxLayer<T>> masked_softmax_layer_;
  std::unique_ptr<SoftmaxLayer<T>> softmax_layer_;

  std::vector<T> debug_vector_;
};

}  // namespace HugeCTR

/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/layer.hpp"

#include <set>
#include <vector>

namespace HugeCTR {

/**
 * Layer which splits a single 2D input tensor into multiple 2D output tensors across columns.
 * e.g., (batch_size, 90) to (batch_size, 40) and (batch_size, 4) by choosing the column ranges
 * [0:40) and (50:90). It is possible those ranges overlap, e.g., [0:100) and [50:200).
 */
class SliceLayer : public Layer {
 public:
  /**
   * Ctor of SliceLayer.
   * @param in_tensor input tensor
   * @param out_tensor vector where the pointers to the created output tensors are stored
   * @param blobs_buff GeneralBuffer used to create the output tensor
   * @param ranges set of the slice ranges along columns
   * @param device_id the id of GPU where this layer belongs
   */
  SliceLayer(const std::shared_ptr<Tensor<float>>& in_tensor,
             Tensors<float>& out_tensors,
             const std::shared_ptr<GeneralBuffer<float>>& blobs_buff,
             std::vector<std::pair<int,int>>& ranges,
             int device_id);
  ~SliceLayer() override {};

  /**
   * Slice's foward pass to gather data to the output tensor
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * Slice's backward pass to scatter data to the input tensors
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

  template <typename T>
  struct OutParam {
    T* out;
    const int st;
    const int ed;
  };

 private:
  void prop_common(bool forward, cudaStream_t stream);
  std::vector<OutParam<float>> set_out_params(int n);
  template <typename... Args>
  void kernel_launch(bool forward, cudaStream_t stream, Args&... args);

  int n_sms_;
  int virt_w_;
  std::vector<int> sts_;
};

}  // namespace HugeCTR

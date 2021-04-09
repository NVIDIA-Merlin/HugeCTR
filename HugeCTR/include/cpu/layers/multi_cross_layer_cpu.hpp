/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <functional>
#include <vector>
#include <cpu/layer_cpu.hpp>

namespace HugeCTR {

class MultiCrossLayerCPU : public LayerCPU {
 private:
  const int num_layers_;
  Tensors2<float> blob_tensors_; /**< vector of internal blobs' tensors */
  Tensors2<float> vec_tensors_;  //[h,1]

  Tensor2<float> tmp_mat_tensors_[3];  //[h,w]
  Tensor2<float> tmp_vec_tensor_;      //[h,1]

  /*
   * stores the weight tensors of this layer.
   */
  // Tensors<float> weights_; It is inherited from Layer, and named as weights_;
  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors2<float> wgrad_;
  /*
   * stores the references to the input tensors of this layer.
   */
  Tensors2<float> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<float> out_tensors_;

 public:
  /**
   * forward pass
   */
  void fprop(bool is_train) final;
  /**
   * backward pass
   */
  void bprop() final;

  MultiCrossLayerCPU(const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                  const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                  const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff,
                  const Tensor2<float>& in_tensor, const Tensor2<float>& out_tensor,
                  int num_layers);
  MultiCrossLayerCPU(const MultiCrossLayerCPU&) = delete;
  MultiCrossLayerCPU& operator=(const MultiCrossLayerCPU&) = delete;

};
}  // namespace HugeCTR

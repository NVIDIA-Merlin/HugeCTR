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

#include <layer.hpp>

namespace HugeCTR {

/**
 * The order2 expression in FM formular(reference paper:
 * https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf).
 * The layer will be used in DeepFM model to implement the FM order2
 * computation (reference code implemented in Tensorflow: line 92~104,
 * https://github.com/ChenglongChen/tensorflow-DeepFM/blob/master/DeepFM.py).
 */
class FmOrder2Layer : public Layer {
  /*
   * stores the weight tensors of this layer.
   */
  Tensors2<float> weights_;
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
   * Ctor of FmOrder2Layer.
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  FmOrder2Layer(const Tensor2<float>& in_tensor, const Tensor2<float>& out_tensor,
                const std::shared_ptr<GPUResource>& gpu_resource);

  /**
   * A method of implementing the forward pass of FmOrder2
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train);

  /**
   * A method of implementing the backward pass of FmOrder2
   * @param stream CUDA stream where the backward propagation is executed
   */
  void bprop();

 private:
  int batch_size_;
  int slot_num_;
  int embedding_vec_size_;
};

}  // namespace HugeCTR

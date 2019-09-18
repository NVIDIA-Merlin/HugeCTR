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

#include <functional>
#include <vector>
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/tensor.hpp"

namespace HugeCTR {
/**
 * @brief
 *
 * Implements three kinds of loss calculations, cross entropy, binary cross entropy
 * and the multi-class cross-entropy loss.
 *
 * Loss is the base class.
 *
 * The forward and backward passes are fused into one function called fused_loss_computation.
 *
 */
class Loss {
 private:
  const int device_id_;

 protected:
  /**
   * label_tensors_: stores the label information during the training process.
   */
  std::vector<std::reference_wrapper<Tensor<float>>> label_tensors_;
  /**
   * input_tensors_: at beginning, the input_tensors_ stores the result of the last layer for the
   * final loss calculation.
   *
   * After the fused_loss_computation is called, the input_tensors_ will be updated to the input
   * gradiant values for the backward pass.
   */
  std::vector<std::reference_wrapper<Tensor<float>>> input_tensors_;
  /**
   * loss_tensors: contains a single value, which stores the average cross entropy, binary cross
   * entropy or multi-class cross entropy loss value.
   */
  std::vector<std::reference_wrapper<Tensor<float>>> loss_tensors_;

 public:
  /**
   * @brief
   * Forward and backward passes are fused into one function.
   *
   * When WMMA is turned on, the scaler set during the compiling process is multiplied to the loss
   * gradient values to prevent the overflow issue.
   *
   * @param stream CUDA stream where the fused_loss_computation is executed in
   */
  virtual void fused_loss_computation(cudaStream_t stream) = 0;
  /**
   * @param device_id GPU device executed on
   */
  Loss(int device_id) : device_id_(device_id) {}
  Loss(const Loss& C) = delete;
  Loss& operator=(const Loss& C) = delete;
  int get_device_id() const { return device_id_; }
  virtual ~Loss() {}
};

class CrossEntropyLoss : public Loss {
 public:
  void fused_loss_computation(cudaStream_t stream) final;
  CrossEntropyLoss(Tensor<float>& label_tensors, Tensor<float>& input_tensors,
                   Tensor<float>& loss_tensors, int device_id);
};

class BinaryCrossEntropyLoss : public Loss {
 public:
  void fused_loss_computation(cudaStream_t stream) final;
  BinaryCrossEntropyLoss(Tensor<float>& label_tensors, Tensor<float>& input_tensors,
                         Tensor<float>& loss_tensors, int device_id);
};

class MultiCrossEntropyLoss : public Loss {
 private:
  GeneralBuffer<float>* internal_buff_;
  Tensor<float>* target_weight_;

 public:
  void fused_loss_computation(cudaStream_t stream) final;
  MultiCrossEntropyLoss(Tensor<float>& label_tensor, Tensor<float>& input_tensor,
                        Tensor<float>& loss_tensor, const std::vector<float> target_weight,
                        int device_id);
};

}  // namespace HugeCTR

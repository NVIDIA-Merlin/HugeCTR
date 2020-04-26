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
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/regularizer.hpp"
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
 public:
  /**
   * @brief
   * Forward and backward passes are fused into one function.
   *
   * gradient values to prevent the overflow issue.
   *
   * @param stream CUDA stream where the fused_loss_computation is executed in
   */
  virtual void fused_loss_computation(cudaStream_t stream);

  /**
   * @param device_id GPU device executed on
   */
  Loss(const std::shared_ptr<const Tensor<float>>& label_tensor,
       const std::shared_ptr<Tensor<float>>& input_tensor,
       const std::shared_ptr<Tensor<float>>& loss_tensor,
       const std::shared_ptr<Regularizer> regularizer, int device_id, int total_gpu_count, float scaler = 1.0);
  Loss(const Loss& C) = delete;
  Loss& operator=(const Loss& C) = delete;
  virtual ~Loss() {}

  int get_device_id() const { return device_id_; }

 protected:
  /**
   * label_tensors_: stores the label information during the training process.
   */
  std::vector<std::shared_ptr<const Tensor<float>>> label_tensors_;
  /**
   * input_tensors_: at beginning, the input_tensors_ stores the result of the last layer for the
   * final loss calculation.
   *
   * After the fused_loss_computation is called, the input_tensors_ will be updated to the input
   * gradiant values for the backward pass.
   */
  std::vector<std::shared_ptr<Tensor<float>>> input_tensors_;
  /**
   * loss_tensors: contains a single value, which stores the average cross entropy, binary cross
   * entropy or multi-class cross entropy loss value.
   */
  std::vector<std::shared_ptr<Tensor<float>>> loss_tensors_;


  int total_gpu_count_;
 private:
  virtual void do_fused_loss_computation(float* input, const float* label, float* loss,
                                         int batch_size, int feature_dim, float scaler, float rterm,
                                         cudaStream_t stream) = 0;
  float scaler_;
  std::shared_ptr<Regularizer> regularizer_;
  const int device_id_;
};

class CrossEntropyLoss : public Loss {
 public:
  void do_fused_loss_computation(float* input, const float* label, float* loss, int batch_size,
                                 int feature_dim, float scaler, float rterm,
                                 cudaStream_t stream) override final;
  CrossEntropyLoss(const std::shared_ptr<const Tensor<float>>& label_tensor,
                   const std::shared_ptr<Tensor<float>>& input_tensor,
                   const std::shared_ptr<Tensor<float>>& loss_tensor,
                   const std::shared_ptr<Regularizer> regularizer, int device_id, int total_gpu_count, float scaler = 1.f);
};

class BinaryCrossEntropyLoss : public Loss {
 public:
  void do_fused_loss_computation(float* input, const float* label, float* loss, int batch_size,
                                 int feature_dim, float scaler, float rterm,
                                 cudaStream_t stream) override final;
  BinaryCrossEntropyLoss(const std::shared_ptr<const Tensor<float>>& label_tensor,
                         const std::shared_ptr<Tensor<float>>& input_tensor,
                         const std::shared_ptr<Tensor<float>>& loss_tensor,
                         const std::shared_ptr<Regularizer> regularizer, int device_id, int total_gpu_count, float scaler = 1.f);
};

class MultiCrossEntropyLoss : public Loss {
 private:
  std::shared_ptr<GeneralBuffer<float>> internal_buff_;
  std::unique_ptr<Tensor<float>> target_weight_;

 public:
  void do_fused_loss_computation(float* input, const float* label, float* loss, int batch_size,
                                 int feature_dim, float scaler, float rterm,
                                 cudaStream_t stream) override final;
  MultiCrossEntropyLoss(const std::shared_ptr<const Tensor<float>>& label_tensor,
                        const std::shared_ptr<Tensor<float>>& input_tensor,
                        const std::shared_ptr<Tensor<float>>& loss_tensor,
                        const std::shared_ptr<Regularizer> regularizer,
                        const std::vector<float>& target_weight, int device_id, int total_gpu_count, float scaler = 1.f);
};

}  // namespace HugeCTR

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
#include <general_buffer2.hpp>
#include <regularizer.hpp>
#include <vector>

namespace HugeCTR {

class ILoss {
 public:
  virtual void compute(bool is_train, cudaStream_t stream) = 0;
  virtual int get_device_id() const = 0;
};

/**
 * @brief
 *
 * Implements three kinds of loss calculations, cross entropy, binary cross entropy
 * and the multi-class cross-entropy loss.
 *
 * Loss is the base class.
 *
 * The forward and backward passes are fused into one function called compute.
 *
 */
template <typename T>
class Loss : public ILoss {
  std::shared_ptr<Regularizer<T>> regularizer_;
  const int device_id_;
  int total_gpu_count_;
  float scaler_;

  /**
   * label_tensors_: stores the label information during the training process.
   */
  Tensors2<float> train_label_tensors_;
  Tensors2<float> evaluate_label_tensors_;
  /**
   * input_tensors_: at beginning, the input_tensors_ stores the result of the last layer for the
   * final loss calculation.
   *
   * After the train is called, the input_tensors_ will be updated to the input
   * gradiant values for the backward pass.
   */
  Tensors2<T> train_input_tensors_;
  Tensors2<T> evaluate_input_tensors_;
  /**
   * loss_tensors: contains a single value, which stores the average cross entropy, binary cross
   * entropy or multi-class cross entropy loss value.
   */
  Tensors2<float> loss_tensors_;

  virtual void do_compute(T* input, const float* label, float* loss, int batch_size,
                          int feature_dim, float scaler, float rterm, bool is_train,
                          cudaStream_t stream) = 0;

  const Tensors2<float>& get_label_tensors(bool is_train) const {
    if (is_train) {
      return train_label_tensors_;
    } else {
      return evaluate_label_tensors_;
    }
  }

  Tensors2<T>& get_input_tensors(bool is_train) {
    if (is_train) {
      return train_input_tensors_;
    } else {
      return evaluate_input_tensors_;
    }
  }

 protected:
  const Tensors2<float>& get_loss_tensors() { return loss_tensors_; }

 public:
  /**
   * @brief
   * Forward and backward passes are fused into one function.
   *
   * gradient values to prevent the overflow issue.
   *
   * @param is_train whether it is on training or evaluating
   * @param stream CUDA stream where the train is executed in
   */
  void compute(bool is_train, cudaStream_t stream) override;

  /**
   * @param device_id GPU device executed on
   */
  Loss(const Tensor2<float>& train_label_tensor, const Tensor2<T>& train_input_tensor,
       const Tensor2<float>& evaluate_label_tensor, const Tensor2<T>& evaluate_input_tensor,
       const Tensor2<float>& loss_tensor, const std::shared_ptr<Regularizer<T>> regularizer,
       int device_id, int total_gpu_count, float scaler = 1.0);
  Loss(const Loss&) = delete;
  Loss& operator=(const Loss&) = delete;
  virtual ~Loss() {}

  int get_device_id() const override { return device_id_; }

 protected:
  int get_total_gpu_count() const { return total_gpu_count_; }
};

template <typename T>
class CrossEntropyLoss : public Loss<T> {
 public:
  void do_compute(T* input, const float* label, float* loss, int batch_size, int feature_dim,
                  float scaler, float rterm, bool is_train, cudaStream_t stream) override final;
  CrossEntropyLoss(const Tensor2<float>& label_tensor, const Tensor2<T>& input_tensor,
                   const Tensor2<float>& loss_tensor,
                   const std::shared_ptr<Regularizer<T>> regularizer, int device_id,
                   int total_gpu_count, float scaler = 1.f);
};

template <typename T>
class BinaryCrossEntropyLoss : public Loss<T> {
 public:
  void do_compute(T* input, const float* label, float* loss, int batch_size, int feature_dim,
                  float scaler, float rterm, bool is_train, cudaStream_t stream) override final;
  BinaryCrossEntropyLoss(const Tensor2<float>& train_label_tensor,
                         const Tensor2<T>& train_input_tensor,
                         const Tensor2<float>& evaluate_label_tensor,
                         const Tensor2<T>& evaluate_input_tensor, const Tensor2<float>& loss_tensor,
                         const std::shared_ptr<Regularizer<T>> regularizer, int device_id,
                         int total_gpu_count, float scaler = 1.f);
};

template <typename T>
class MultiCrossEntropyLoss : public Loss<T> {
 private:
  Tensor2<float> target_weight_;

 public:
  void do_compute(T* input, const float* label, float* loss, int batch_size, int feature_dim,
                  float scaler, float rterm, bool is_train, cudaStream_t stream) override final;
  MultiCrossEntropyLoss(const Tensor2<float>& label_tensor, const Tensor2<T>& input_tensor,
                        const Tensor2<float>& loss_tensor,
                        const std::shared_ptr<Regularizer<T>> regularizer,
                        const std::vector<float>& target_weight, int device_id, int total_gpu_count,
                        float scaler = 1.f);
};

}  // namespace HugeCTR

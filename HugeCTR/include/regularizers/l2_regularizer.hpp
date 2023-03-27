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

#include <common.hpp>
#include <core23/tensor_container.hpp>
#include <memory>
#include <regularizer.hpp>
#include <utils.hpp>

namespace HugeCTR {

/**
 * @brief L2 Regularizer
 */
template <typename T>
class L2Regularizer : public Regularizer<T> {
 public:
  /*
   * Constructor of L2Regularizer
   * @param weight_buff GeneralBuffer containing all the layers' weights
   * @param wgrad_buff GeneralBuffer containing all the layers' wgrads
   * @param batch_size Network batch size
   * @param lambda Hyperparameter to scale the regularization term
   * @param cublas_handle cuBLAS handle to execute the kernel in fprop
   * @param device_id Device to be used
   */
  L2Regularizer(const Tensor2<float>& weight_buff, const Tensor2<T>& wgrad_buff,
                const int batch_size, const float lambda,
                const std::shared_ptr<GPUResource>& gpu_resource);

  /*
   * Constructor of L2Regularizer
   * @param weight_tensors TensorContainer of all the layers' weights
   * @param wgrad_tensors TensorContainer of all the layers' wgrads
   * @param batch_size Network batch size
   * @param lambda Hyperparameter to scale the regularization term
   * @param cublas_handle cuBLAS handle to execute the kernel in fprop
   * @param device_id Device to be used
   */
  L2Regularizer(std::optional<WeightTensors> weight_tensors,
                std::optional<WgradTensors<T>> wgrad_tensors, const int batch_size,
                const float lambda, const std::shared_ptr<GPUResource>& gpu_resource);

  /*
   * Destructor of L2Regularizer
   */
  ~L2Regularizer() override {}

 private:
  /*
   * Calculate rterm based on the dot product and scale it with lambda / (2 * batch_size)
   * @param weight the device buffer of weight
   * @param h_rterm the host pointer to the regularization term
   * @param num_elements the number of weight values across layers
   * @param stream CUDA Stream where the kernel is executed
   */
  void do_compute_rterm(const float* weight, float* h_rterm, int num_elements) override;
  /*
   * Initialize wgrad with weight * (lambda / batch_size)
   * @param weight the device buffer of weight
   * @param h_rterm the host pointer to the regularization term
   * @param num_elements the number of weight values across layers
   * @param stream CUDA Stream where the kernel is executed
   */
  void do_initialize_wgrad(const float* weight, T* wgrad, int num_elements,
                           cudaStream_t stream) override;

  const float lambda_;
};

}  // namespace HugeCTR

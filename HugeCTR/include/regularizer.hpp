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
#include <general_buffer2.hpp>
#include <gpu_resource.hpp>
#include <memory>
#include <network_helpers.hpp>
#include <optional>
#include <utils.hpp>

namespace HugeCTR {

/**
 * @brief Abstract base class of Regularizer
 */
template <typename T = float>
class Regularizer {
 public:
  /*
   * Constructor of Regularizer
   * @param weight_buff GeneralBuffer containing all the layers' weights
   * @param wgrad_buff GeneralBuffer containing all the layers' wgrads
   * @param batch_size Network batch size
   * @param device_id Device to be used
   */
  Regularizer(const Tensor2<float>& weight_buff, const Tensor2<T>& wgrad_buff, const int batch_size,
              const std::shared_ptr<GPUResource>& gpu_resource);

  /*
   * Constructor of Regularizer
   * @param weight_tensors TensorContainer of all the layers' weights
   * @param wgrad_tensors TensorContainer of all the layers' wgrads
   * @param batch_size Network batch size
   * @param device_id Device to be used
   */
  Regularizer(std::optional<WeightTensors> weight_tensors,
              std::optional<WgradTensors<T>> wgrad_tensors, const int batch_size,
              const std::shared_ptr<GPUResource>& gpu_resource);

  /*
   * Destructor of Regularizer
   */
  virtual ~Regularizer() = default;

  virtual void initialize() {}

  /*
   * Function that computes the regularization term
   * To customize it, override th private function do_compute_rterm
   * @param stream CUDA Stream where the kernel is executed
   */
  void compute_rterm();

  /*
   * Function that initialize wgrad
   * To customize it, override th private function do_initialize_wgrad
   * @param stream CUDA Stream where the kernel is executed
   */
  void initialize_wgrad();

  /*
   * Return the calculated regularization term
   */
  float get_rterm() const { return h_rterm_; }

 protected:
  int get_batch_size() const { return batch_size_; }
  int get_device_id() const { return gpu_resource_->get_device_id(); }
  const GPUResource& get_gpu() const { return *gpu_resource_; }

 private:
  /*
   * To compute the regularization term, override this function.
   * It is called inside the public function compute_rterm
   * @param weight the device buffer of weight
   * @param h_rterm the host pointer to the regularization term
   * @param num_elements the number of weight values across layers
   * @param stream CUDA Stream where the kernel is executed
   */
  virtual void do_compute_rterm(const float* weight, float* h_rterm, int num_elements) = 0;
  /*
   * To initialize wgrad, override this function.
   * It is called inside the public function initialize_wgrad
   * @param weight the device buffer of weight
   * @param wgrad the device buffer of wgrad
   * @param num_elements the number of weight values across layers
   * @param stream CUDA Stream where the kernel is executed
   */
  virtual void do_initialize_wgrad(const float* weight, T* wgrad, int num_elements,
                                   cudaStream_t stream) = 0;

  Tensor2<float> weight_buff_;
  Tensor2<T> wgrad_buff_;
  std::optional<WeightTensors> weight_tensors_;
  std::optional<WgradTensors<T>> wgrad_tensors_;
  int batch_size_;
  float h_rterm_;
  std::shared_ptr<GPUResource> gpu_resource_;
};

}  // namespace HugeCTR

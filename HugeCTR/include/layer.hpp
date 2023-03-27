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

#include <core23/tensor.hpp>
#include <cpu_resource.hpp>
#include <data_simulator.hpp>
#include <fstream>
#include <functional>
#include <general_buffer2.hpp>
#include <gpu_resource.hpp>
#include <string>
#include <vector>

namespace HugeCTR {

/**
 * @brief
 * Definition of a basic layer class.
 */
class Layer {
 protected:
  std::vector<core23::Tensor> input_tensors_;
  std::vector<core23::Tensor> output_tensors_;

 private:
  /*
   * Specify which GPU device will be executed on.
   */
  std::shared_ptr<GPUResource> gpu_resource_;

 protected:
  /*
   * stores the initializer types of this layer.
   */
  std::vector<Initializer_t> initializer_types_;
  /*
   * stores the weight tensors of this layer.
   */
  Tensors2<float> weights_;

  const GPUResource& get_gpu() const { return *gpu_resource_; }
  int get_device_id() const { return gpu_resource_->get_device_id(); }

 public:
  /*
   * Forward pass
   * @param stream: the CUDA stream that the forward function will be executed on.
   */
  virtual void fprop(bool is_train) = 0;
  /*
   * Backward pass
   * @param stream: the CUDA stream that the forward function will be executed on.
   */
  virtual void bprop() = 0;

  virtual std::string get_no_trained_params_in_string() { return std::string(); }

  virtual std::vector<TensorBag2> get_tensors_for_non_trainable_params() {
    return std::vector<TensorBag2>();
  }

  virtual std::vector<core23::Tensor> get_non_trainable_params_as_tensors() {
    return std::vector<core23::Tensor>();
  }

  virtual void init_params(const curandGenerator_t& generator);

  Layer(const std::vector<core23::Tensor>& input_tensors,
        const std::vector<core23::Tensor>& output_tensors,
        const std::shared_ptr<GPUResource>& gpu_resource,
        std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>());
  Layer(const std::shared_ptr<GPUResource>& gpu_resource,
        std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>())
      : Layer({}, {}, gpu_resource, initializer_types) {}
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;
  virtual ~Layer() = default;

  /*
   * Some of the layers requires initialize like fully connected layer
   */
  virtual void initialize() {}

  /*
   * Some of the layers requires algorithm search like fully connected layer
   */
  virtual void search_algorithm() {}

 private:
  /*
   * Layer initializer. If a layer wants the specific weight initialization,
   * Override each private function accordingly, e.g., BatchNormLayer
   */
  virtual std::unique_ptr<DataSimulator> get_zero_initializer(const int index) {
    return std::make_unique<ConstantDataSimulator>(0.0f);
  }

  virtual std::unique_ptr<DataSimulator> get_uniform_initializer(const int index) {
    return std::move(get_default_initializer(index));
  }
  virtual std::unique_ptr<DataSimulator> get_xavier_uniform_initializer(const int index) {
    return std::move(get_default_initializer(index));
  }
  virtual std::unique_ptr<DataSimulator> get_xavier_norm_initializer(const int index) {
    return std::move(get_default_initializer(index));
  }
  virtual std::unique_ptr<DataSimulator> get_default_initializer(const int index) {
    return std::move(get_zero_initializer(index));
  }
};

}  // namespace HugeCTR

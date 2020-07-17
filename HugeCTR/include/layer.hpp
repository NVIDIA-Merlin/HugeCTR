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

#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "HugeCTR/include/tensor.hpp"

namespace HugeCTR {
/**
 * @brief
 * Definition of a basic layer class.
 */
class Layer {
 private:
  /*
   * Specify which GPU device will be executed on.
   */
  const int device_id_;

 protected:
  /*
   * stores the initializer types of this layer.
   */
  std::vector<Initializer_t> initializer_types_;
  /*
   * stores the weight tensors of this layer.
   */
  Tensors<float> weights_;

 public:
  /*
   * Forward pass
   * @param stream: the CUDA stream that the forward function will be executed on.
   */
  virtual void fprop(cudaStream_t stream) = 0;
  /*
   * Backward pass
   * @param stream: the CUDA stream that the forward function will be executed on.
   */
  virtual void bprop(cudaStream_t stream) = 0;
  /*
   * Inference pass (most layers just call fprop but some layer like dropout should inherit it)
   * @param stream: the CUDA stream that the forward function will be executed on.
   */
  virtual void inference(cudaStream_t stream) { fprop(stream); }

  virtual std::string get_no_trained_params_in_string() { return std::string(); }
  void init_params(std::ofstream& out_stream);
  inline int get_device_id() const { return device_id_; }
  Layer(int device_id, std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>()) 
    : device_id_(device_id), initializer_types_(initializer_types) {}
  Layer(const Layer& C) = delete;
  Layer& operator=(const Layer& C) = delete;
  virtual ~Layer() {}
  /*
   * Some of the layers requires algorithm search like fully connected layer
   */
  virtual void optimize() {}

  std::vector<float> get_initializer();

 private:
 
/*
* Layer initializer. If a layer wants the specific weight initialization,
* Override each private function accordingly, e.g., BatchNormLayer
*/
  std::unique_ptr<DataSimulator<float>> get_zero_initializer(const int index) {
    auto zero_init = [] {return static_cast<float>(0); };
    return std::unique_ptr<DataSimulator<float>>(new SingleDataSimulator<float>(zero_init));
  }
  
  virtual std::unique_ptr<DataSimulator<float>> get_uniform_initializer(const int index) {
    return std::move(get_default_initializer(index));
  }
  virtual std::unique_ptr<DataSimulator<float>> get_xavier_uniform_initializer(const int index) {
    return std::move(get_default_initializer(index));
  }
  virtual std::unique_ptr<DataSimulator<float>> get_xavier_norm_initializer(const int index) {
    return std::move(get_default_initializer(index));
  }
  virtual std::unique_ptr<DataSimulator<float>> get_default_initializer(const int index) {
    return std::move(get_zero_initializer(index));
  }
  

};
}  // namespace HugeCTR

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
   * stores the weight tensors of this layer.
   */
  std::vector<Tensor<float>*> weights_;
  /*
   * stores the weight gradient tensors of this layer.
   */
  std::vector<Tensor<float>*> wgrad_;
  /*
   * stores the references to the input tensors of this layer.
   */
  std::vector<std::reference_wrapper<Tensor<float>>> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  std::vector<std::reference_wrapper<Tensor<float>>> out_tensors_;

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
  virtual std::string get_no_trained_params_in_string() { return std::string(); }
  void init_params(std::ofstream& out_stream);
  virtual std::vector<std::reference_wrapper<Tensor<float>>> get_in_tensor() { 
    return in_tensors_;
  }
  inline int get_device_id() const { return device_id_; }
  // Layer(GeneralBuffer& weight_buff, GeneralBuffer& wgrad_buff, int device_id); need to implement
  // this in children
  Layer(int device_id) : device_id_(device_id) {}
  Layer(const Layer& C) = delete;
  Layer& operator=(const Layer& C) = delete;
  virtual ~Layer();

 private:
  virtual std::vector<float> get_initializer() { return std::vector<float>(); }
};
}  // namespace HugeCTR

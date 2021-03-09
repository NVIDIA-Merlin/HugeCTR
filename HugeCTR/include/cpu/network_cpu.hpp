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

#include <common.hpp>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <vector>

#include <cpu/layer_cpu.hpp>
#include <parser.hpp>

namespace HugeCTR {

/**
 * @brief Dense network (embedding is not included)
 *
 * Each GPU (device) has an instance of Network. Network performs
 * forward/backward/loss/update of the dense layers.
 */
class NetworkCPU {
 private:
  std::vector<std::unique_ptr<LayerCPU>> layers_;    /**< vector of layers */

  Tensor2<float> weight_tensor_;
  Tensor2<float> wgrad_tensor_;
  Tensor2<__half> weight_tensor_half_;
  Tensor2<__half> wgrad_tensor_half_;

  Tensor2<float> pred_tensor_;

  std::shared_ptr<CPUResource> cpu_resource_;
  // std::shared_ptr<GPUResource> gpu_resource_; /**< gpu resource */

  bool use_mixed_precision_;
  // bool enable_cuda_graph_;

  // bool predict_graph_created_;
  // bool eval_graph_created_;
  // bool train_fprop_graph_created_;
  // bool train_bprop_graph_created_;
  // cudaGraph_t predict_graph_;
  // cudaGraph_t eval_graph_;
  // cudaGraph_t train_fprop_graph_;
  // cudaGraph_t train_bprop_graph_;
  // cudaGraphExec_t predict_instance_;
  // cudaGraphExec_t eval_instance_;
  // cudaGraphExec_t train_fprop_instance_;
  // cudaGraphExec_t train_bprop_instance_;

  void conv_weight_(Tensor2<__half>& target, const Tensor2<float>& source);

 public:
  /**
   * Ctor.
   * @param device_id device id.
   * @param gpu_resource gpu resource for local gpu.
   * @param disable_parser only for unit test.
   */
  NetworkCPU(const std::shared_ptr<CPUResource>& cpu_resource, 
          bool use_mixed_precision = false);
  NetworkCPU(const NetworkCPU&) = delete;
  NetworkCPU& operator=(const NetworkCPU&) = delete;

  /**
   * Forward only for inference.
   */
  void predict();

  /**
   * Get the pred tensor for inference.
   */
  Tensor2<float> get_pred_tensor() { return pred_tensor_; }

  /**
   * Get number of parameters in this network.
   */
  size_t get_params_num() const { return weight_tensor_.get_num_elements(); }


  /**
   * Read parameters from model_file.
   */
  void load_params_from_model(const std::string& model_file);

  /**
   * initialize layer by layer
   */
  void initialize();

  /**
   * factory method to create network
   */
  static NetworkCPU* create_network(const nlohmann::json& j_array,
                                 std::vector<TensorEntry>& tensor_entries,
                                 const std::shared_ptr<CPUResource>& cpu_resource,
                                 bool use_mixed_precision);
};

}  // namespace HugeCTR

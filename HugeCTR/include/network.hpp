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

#include <cublas_v2.h>
#include <nccl.h>
#include <common.hpp>
#include <fstream>
#include <functional>
#include <gpu_resource.hpp>
#include <layer.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <nlohmann/json.hpp>
#include <optimizer.hpp>
#include <vector>

namespace HugeCTR {

enum class TensorUse { General, Train, Evaluate };

struct TensorEntry {
  std::string name;
  TensorUse use;
  TensorBag2 bag;
};

/**
 * @brief Dense network (embedding is not included)
 *
 * Each GPU (device) has an instance of Network. Network performs
 * forward/backward/loss/update of the dense layers.
 */
class Network {
  friend Network* create_network(const nlohmann::json& j_array, const nlohmann::json& j_optimizor,
                                 std::vector<TensorEntry>& tensor_entries,
                                 int num_networks_in_global,
                                 const std::shared_ptr<CPUResource>& cpu_resource,
                                 const std::shared_ptr<GPUResource>& gpu_resource,
                                 bool use_mixed_precision,
                                 float scaler,
                                 bool use_algorithm_search,
                                 bool use_cuda_graph);

 private:
  std::vector<std::unique_ptr<Layer>> layers_; /**< vector of layers */

  Tensor2<float> weight_tensor_;
  Tensor2<float> wgrad_tensor_;
  Tensor2<__half> weight_tensor_half_;
  Tensor2<__half> wgrad_tensor_half_;

  std::shared_ptr<CPUResource> cpu_resource_;
  std::shared_ptr<GPUResource> gpu_resource_; /**< gpu resource */
  std::unique_ptr<Optimizer> optimizer_;      /**< optimizer */
  std::unique_ptr<ILoss> loss_;               /**< loss */
  Tensor2<float> loss_tensor_;                /**< loss tensor */
  bool full_fp16_{false};
  bool enable_cuda_graph_;
  void conv_weight_();
  metrics::RawMetricMap raw_metrics_;

  bool eval_graph_created_;
  bool train_fprop_graph_created_;
  bool train_bprop_graph_created_;
  cudaGraph_t eval_graph_;
  cudaGraph_t train_fprop_graph_;
  cudaGraph_t train_bprop_graph_;
  cudaGraphExec_t eval_instance_;
  cudaGraphExec_t train_fprop_instance_;
  cudaGraphExec_t train_bprop_instance_;

 public:
  /**
   * Ctor.
   * @param device_id device id.
   * @param gpu_resource gpu resource for local gpu.
   * @param disable_parser only for unit test.
   */
  Network(const std::shared_ptr<CPUResource>& cpu_resource,
          const std::shared_ptr<GPUResource>& gpu_resource,
          bool full_fp16 = false,
          bool use_cuda_graph = true);
  Network(const Network&) = delete;
  Network& operator=(const Network&) = delete;

  /**
   * Forward, backward and update the network.
   */
  void train(long long current_batchsize);

  /**
   * Forward only.
   */
  void eval();

  /**
   * Get current loss and return.
   */
  float get_loss();

  int get_device_id() const { return gpu_resource_->get_device_id(); }

  metrics::RawMetricMap get_raw_metrics() const;

  /**
   * Get number of parameters in this network.
   */
  size_t get_params_num() const { return weight_tensor_.get_num_elements(); }

  /**
   * Writting paramters to fstream.
   */
  void download_params_to_host(std::ofstream& weight_stream);

  /**
   * Get no trained parameters (such as parameters in Batch nomalization) to string.
   */
  std::string get_no_trained_params_in_string();

  /**
   * Read parameters from model_file.
   */
  void upload_params_to_device(const std::string& model_file);

  /**
   * Writting paramters to cpu buffer.
   */
  void download_params_to_host(float* weight);

  /**
   * Read parameters from cpu buffer.
   */
  void upload_params_to_device(float* params);

  /**
   * Init parameters and write to fstream.
   */
  void init_params(const std::string& model_file_name);

  /**
   * Copy parameters from a network.
   */
  void copy_params(const Network& n);

  /**
   * Exchange wgrad between gpus.
   */
  void exchange_wgrad();

  /**
   * Update parameters.
   */
  void update_params();

  /**
   * reset the learning rate to lr.
   */
  void set_learning_rate(float lr) { optimizer_->set_learning_rate(lr); }

  /**
   * initialize layer by layer
   */
  void initialize() {
    for (auto& n : layers_) {
      n->initialize();
    }
  }

  /**
   * search_algorithm layer by layer
   */
  void search_algorithm() {
    for (auto& n : layers_) {
      n->search_algorithm();
    }
  }
};  // namespace HugeCTR

}  // namespace HugeCTR

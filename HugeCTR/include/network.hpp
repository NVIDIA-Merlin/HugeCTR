/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <exchange_wgrad.hpp>
#include <fstream>
#include <functional>
#include <gpu_resource.hpp>
#include <graph_wrapper.hpp>
#include <hdfs_backend.hpp>
#include <layer.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <nlohmann/json.hpp>
#include <optimizer.hpp>
#include <vector>

namespace HugeCTR {

struct DenseLayer;
class Model;

struct TensorEntry {
  std::string name;
  TensorBag2 bag;
};

/**
 * @brief Dense network (embedding is not included)
 *
 * Each GPU (device) has an instance of Network. Network performs
 * forward/backward/loss/update of the dense layers.
 */
class Network {
 protected:
  std::vector<std::unique_ptr<Layer>> train_layers_;    /**< vector of layers */
  std::vector<std::unique_ptr<Layer>> evaluate_layers_; /**< vector of layers */

  std::map<std::string, std::unique_ptr<ILoss>> train_losses_;    /**< map of loss layers */
  std::map<std::string, std::unique_ptr<ILoss>> evaluate_losses_; /**< map of loss layers */
  std::map<std::string, float> loss_weights_;                     /** < map of weights for losses */

  std::map<std::string, int> label_dims_; /** < map of dimensions of labels */

  std::unique_ptr<Optimizer> optimizer_; /**< optimizer */
  std::vector<Layer*> top_layers_, bottom_layers_;

  Tensor2<float> train_weight_tensor_;
  Tensor2<float> wgrad_tensor_;
  Tensor2<float> evaluate_weight_tensor_;
  Tensor2<float> opt_tensor_;
  Tensor2<__half> train_weight_tensor_half_;
  Tensor2<__half> wgrad_tensor_half_;
  Tensor2<__half> evaluate_weight_tensor_half_;
  Tensor2<__half> opt_tensor_half_;

  std::map<std::string, Tensor2<float>> train_loss_tensors_;    /**< map of loss tensors */
  std::map<std::string, Tensor2<float>> evaluate_loss_tensors_; /**< map of loss tensor */

  metrics::RawMetricMap raw_metrics_;

  Tensor2<float> pred_tensor_;
  Tensor2<__half> pred_tensor_half_;

  std::shared_ptr<CPUResource> cpu_resource_;
  std::shared_ptr<GPUResource> gpu_resource_; /**< gpu resource */
  bool use_mixed_precision_;
  bool enable_cuda_graph_;

  GraphWrapper predict_graph_, eval_graph_, train_fprop_graph_, train_bprop_graph_;
  GraphWrapper bottom_train_fprop_graph_, bottom_train_bprop_graph_;

  void conv_weight_(Tensor2<__half>& target, const Tensor2<float>& source);

  std::map<TrainState_t, cudaEvent_t> train_events_;
  std::shared_ptr<GpuLearningRateScheduler> lr_sched_;

  template <typename LPtr>
  void prop_layers(const std::vector<LPtr>& layers, GraphWrapper& graph, bool use_graph, bool fprop,
                   const cudaStream_t stream, bool train = true);
  cudaEvent_t& get_train_events(TrainState_t state);

 public:
  /**
   * Ctor.
   * @param device_id device id.
   * @param gpu_resource gpu resource for local gpu.
   * @param disable_parser only for unit test.
   */
  Network(const std::shared_ptr<CPUResource>& cpu_resource,
          const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision = false,
          bool use_cuda_graph = true);
  Network(const Network&) = delete;
  Network& operator=(const Network&) = delete;

  /**
   * Forward, backward and update the network.
   */
  virtual void train(long long current_batchsize);

  virtual TrainState train(long long current_batchsize, std::function<void()> exchange_wgrad,
                           TrainState state);
  /**
   * Forward only.
   */
  virtual void eval(long long current_batchsize);

  /**
   * Forward only for inference.
   */
  void predict();

  /**
   * Get the pred tensor for inference.
   */
  Tensor2<float> get_pred_tensor() { return pred_tensor_; }

  Tensor2<__half> get_pred_tensor_half() { return pred_tensor_half_; }

  /**
   * Get current loss and return.
   */
  float get_loss();

  int get_device_id() const { return gpu_resource_->get_device_id(); }

  metrics::RawMetricMap get_raw_metrics() const;

  /**
   * Get number of parameters in this network.
   */
  size_t get_params_num() const { return train_weight_tensor_.get_num_elements(); }

  size_t get_opt_states_size_in_byte() const {
    return use_mixed_precision_ ? opt_tensor_half_.get_size_in_bytes()
                                : opt_tensor_.get_size_in_bytes();
  }

  /**
   * Writting paramters to fstream.
   */
  void download_params_to_host(std::ofstream& weight_stream);

  /**
   * Writting paramters to HDFS.
   */
  void download_params_to_hdfs(std::string& write_path, DataSourceParams data_source_params);

  /**
   * Writting opt states to fstream.
   */
  void download_opt_states_to_host(std::ofstream& opt_states_stream);

  /**
   * Writting opt states to HDFS.
   */
  void download_opt_states_to_hdfs(std::string& write_path, DataSourceParams data_source_params);

  /**
   * Get no trained parameters (such as parameters in Batch nomalization) to string.
   */
  std::string get_no_trained_params_in_string();

  /**
   * Read parameters from model_file.
   */
  void upload_params_to_device(const std::string& model_file);

  /**
   * Read parameters from model_file.
   */
  void upload_params_to_device_inference(const std::string& model_file);

  /**
   * Writting paramters to cpu buffer.
   */
  void download_params_to_host(float* weight);

  /**
   * Read parameters from cpu buffer.
   */
  void upload_params_to_device(float* params);

  /**
   * Read opt states from cpu buffer.
   */
  void upload_opt_states_to_device(char* h_opt_states);

  /**
   * Init parameters and write to fstream.
   */
  void init_params(size_t index);

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
   * set the learing rate scheduling delegate instead of explicitly setting the learning rate.
   */
  void set_learning_rate_scheduler(std::shared_ptr<GpuLearningRateScheduler>& lr_sched) {
    optimizer_->set_learning_rate_scheduler(lr_sched);
    lr_sched_ = lr_sched;
  }

  /**
   * initialize layer by layer
   */
  void initialize(bool is_train = true);

  /**
   * search_algorithm layer by layer
   */
  void search_algorithm();

  /**
   * factory method to create network
   */
  static Network* create_network(const nlohmann::json& j_array, const nlohmann::json& j_optimizer,
                                 std::vector<TensorEntry>& train_tensor_entries,
                                 std::vector<TensorEntry>& evaluate_tensor_entries,
                                 int num_networks_in_global,
                                 std::shared_ptr<ExchangeWgrad>& exchange_wgrad,
                                 const std::shared_ptr<CPUResource>& cpu_resource,
                                 const std::shared_ptr<GPUResource>& gpu_resource,
                                 bool use_mixed_precision, bool enable_tf32_compute, float scaler,
                                 bool use_algorithm_search, bool use_cuda_graph,
                                 bool inference_flag, bool grouped_all_reduce);

  /**
   * add layer to network, python interface use only
   */
  friend class Model;
  friend class ModelPerfExt;
  /**
   * copy weights from train layers to evaluate layers
   */
  void copy_weights_from_train_layers_to_evaluate_layers();
};

}  // namespace HugeCTR

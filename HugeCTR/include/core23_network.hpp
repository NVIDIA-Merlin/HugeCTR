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

#include <core23/tensor_container.hpp>
#include <cpu_resource.hpp>
#include <gpu_learning_rate_scheduler.hpp>
#include <gpu_resource.hpp>
#include <layer.hpp>
#include <loss.hpp>
#include <map>
#include <memory>
#include <metrics.hpp>
#include <optimizer.hpp>
#include <optional>
#include <string>
#include <vector>

namespace HugeCTR {

/**
 * @brief Dense network (embedding is not included)
 *
 * Each GPU (device) has an instance of Core23TempNetwork. Core23TempNetwork performs
 * forward/backward/loss/update of the dense layers.
 */
class Core23TempNetwork final {
 public:
  /**
   * Ctor.
   * @param device_id device id.
   * @param gpu_resource gpu resource for local gpu.
   * @param disable_parser only for unit test.
   */
  Core23TempNetwork(const std::shared_ptr<CPUResource>& cpu_resource,
                    const std::shared_ptr<GPUResource>& gpu_resource,
                    bool use_mixed_precision = false);
  Core23TempNetwork(const Core23TempNetwork&) = delete;
  Core23TempNetwork& operator=(const Core23TempNetwork&) = delete;
  Core23TempNetwork(Core23TempNetwork&&) = default;
  Core23TempNetwork& operator=(Core23TempNetwork&) = default;

  /**
   * Forward, backward and update the network.
   */
  void train(int64_t current_batchsize);

  /**
   * Forward only.
   */
  void eval(int64_t current_batchsize);

  /**
   * Forward only for inference.
   */
  void predict();

  /**
   * Get the pred tensor for inference.
   */
  core23::Tensor get_pred_tensor() { return pred_tensor_; }

  /**
   * Get current loss and return.
   */
  float get_loss();

  int get_device_id() const { return gpu_resource_->get_device_id(); }

  metrics::Core23MultiLossMetricMap get_raw_metrics_all() const;

  metrics::Core23RawMetricMap get_raw_metrics(std::string) const;

  /**
   * Get number of parameters in this network.
   */
  size_t get_params_num() const { return train_weight_tensor_->num_elements(); }

  size_t get_opt_states_size_in_byte() const { return opt_tensor_->num_bytes(); }

  /**
   * Writting paramters to file.
   */
  void download_params_to_host(const std::string& write_path);

  /**
   * Writting opt states to file.
   */
  void download_opt_states_to_host(const std::string& write_path);

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
   * Read non-trainable parameters from model_file, e.g., running mean and running variable for
   * BatchNorm
   */
  void upload_non_trainable_params_to_device_inference(const std::string& model_file);

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
   * copy weights from train layers to evaluate layers
   */
  void copy_weights_from_train_layers_to_evaluate_layers();

  /**
   * copy non-trainable parameters from train layers to evaluate layers, e.g., running mean and
   * running variance for BatchNorm
   */
  void copy_non_trainable_params_from_train_layers_to_evaluate_layers();

  void set_train_layers(std::vector<std::unique_ptr<Layer>>&& train_layers);
  void set_evaluate_layers(std::vector<std::unique_ptr<Layer>>&& evaluate_layers);
  void set_train_losses(std::map<std::string, std::unique_ptr<ILoss>>&& train_losses,
                        const std::map<std::string, float>& label_weights);
  void set_top_and_bottom_layers(std::vector<Layer*>&& top_layers,
                                 std::vector<Layer*>&& bottom_layers);
  void set_evaluate_losses(std::map<std::string, std::unique_ptr<ILoss>>&& evaluate_losses,
                           const std::map<std::string, float>& label_weights);
  void set_raw_metrics(metrics::Core23MultiLossMetricMap&& raw_metrics);
  void set_optimizer(std::unique_ptr<Optimizer> optimizer);
  void create_and_set_optimizer(const OptParams& opt_params);

 private:
  friend class Model;

  void conv_weight_(std::optional<core23::TensorContainer<__half, 1, 1>>& target_opt,
                    const std::optional<core23::TensorContainer<float, 1, 1>>& source_opt);
  void prop_layers(const std::vector<Layer*>& layers, bool fprop, bool train);

  void set_losses_common(const std::map<std::string, std::unique_ptr<ILoss>>& losses,
                         const std::map<std::string, float>& label_weights,
                         std::map<std::string, core23::Tensor>& loss_tensors);

  std::vector<std::unique_ptr<Layer>> train_layers_;    /**< vector of layers */
  std::vector<std::unique_ptr<Layer>> evaluate_layers_; /**< vector of layers */

  std::map<std::string, std::unique_ptr<ILoss>> train_losses_;    /**< map of loss layers */
  std::map<std::string, std::unique_ptr<ILoss>> evaluate_losses_; /**< map of loss layers */

  std::map<std::string, int> label_dims_; /** < map of dimensions of labels */

  std::unique_ptr<Optimizer> optimizer_; /**< optimizer */
  std::vector<Layer*> top_layers_, bottom_layers_;

  std::optional<core23::TensorContainer<float, 1, 1>> train_weight_tensor_;
  std::optional<core23::TensorContainer<float, 1, 1>> wgrad_tensor_;
  std::optional<core23::TensorContainer<float, 1, 1>> evaluate_weight_tensor_;
  std::optional<core23::TensorContainer<float, 1, 1>> opt_tensor_;
  std::optional<core23::TensorContainer<__half, 1, 1>> train_weight_tensor_half_;
  std::optional<core23::TensorContainer<__half, 1, 1>> wgrad_tensor_half_;
  std::optional<core23::TensorContainer<__half, 1, 1>> evaluate_weight_tensor_half_;

  std::map<std::string, core23::Tensor> train_loss_tensor_;    /**< map of loss tensors */
  std::map<std::string, core23::Tensor> evaluate_loss_tensor_; /**< map of loss tensor */

  metrics::Core23MultiLossMetricMap raw_metrics_; /**< map of metric data for each loss */

  core23::Tensor pred_tensor_;

  std::shared_ptr<CPUResource> cpu_resource_;
  std::shared_ptr<GPUResource> gpu_resource_; /**< gpu resource */
  bool use_mixed_precision_;

  std::shared_ptr<GpuLearningRateScheduler> lr_sched_;
};

}  // namespace HugeCTR

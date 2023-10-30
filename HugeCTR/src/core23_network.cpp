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

#include <omp.h>

#include <core23_network.hpp>
#include <io/filesystem.hpp>
#include <network_helpers.hpp>
#include <nlohmann/json.hpp>
#include <parser.hpp>
#include <regularizer.hpp>
#include <trainable_layer.hpp>

namespace HugeCTR {

void conv_weight_gpu(size_t grid, size_t block, __half* dst, const float* src, int elems,
                     cudaStream_t stream);

Network::Network(const std::shared_ptr<CPUResource>& cpu_resource,
                 const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision)
    : cpu_resource_(cpu_resource),
      gpu_resource_(gpu_resource),
      use_mixed_precision_(use_mixed_precision) {}

// TODO - Update this method for multi-regularzer to run each regularizer and compute the
// conditional weighted sum for each layer.  Then similarly compute the weighted sum of losses
// associated with each layer
void Network::train(int64_t current_batchsize) {
  // forward
  if (use_mixed_precision_) {
    conv_weight_(train_weight_tensor_half_, train_weight_tensor_);
  }

  std::vector<Layer*> train_layers_ptr;
  std::transform(train_layers_.begin(), train_layers_.end(), std::back_inserter(train_layers_ptr),
                 [](const std::unique_ptr<Layer>& layer) { return layer.get(); });
  prop_layers(train_layers_ptr, true, true);

  float rterm = train_losses_.begin()->second->regularizer_compute_rterm();

  for (std::map<std::string, std::unique_ptr<ILoss>>::iterator iter = train_losses_.begin();
       iter != train_losses_.end(); ++iter) {
    iter->second->compute(true, current_batchsize, rterm);
  }

  train_losses_.begin()->second->regularizer_initialize_wgrad(true);  // Only 1 regularizer for now

  prop_layers(train_layers_ptr, false, true);

  return;
}

void Network::eval(int64_t current_batchsize) {
  std::vector<Layer*> evaluate_layers_ptr;
  std::transform(evaluate_layers_.begin(), evaluate_layers_.end(),
                 std::back_inserter(evaluate_layers_ptr),
                 [](const std::unique_ptr<Layer>& layer) { return layer.get(); });
  prop_layers(evaluate_layers_ptr, true, false);

  float rterm = evaluate_losses_.begin()->second->regularizer_compute_rterm();

  for (std::map<std::string, std::unique_ptr<ILoss>>::iterator iter = evaluate_losses_.begin();
       iter != evaluate_losses_.end(); ++iter) {
    iter->second->compute(false, current_batchsize, rterm);
  }

  evaluate_losses_.begin()->second->regularizer_initialize_wgrad(
      false);  // Only 1 regularize for now
}
void Network::predict() {
  std::vector<Layer*> evaluate_layers_ptr;
  std::transform(evaluate_layers_.begin(), evaluate_layers_.end(),
                 std::back_inserter(evaluate_layers_ptr),
                 [](const std::unique_ptr<Layer>& layer) { return layer.get(); });

  prop_layers(evaluate_layers_ptr, true, false);
}

float Network::get_loss() {
  float loss_host = 0.f;
  float* loss_temp = new float[train_loss_tensor_.size()];
  size_t i = 0;

  CudaDeviceContext context(get_device_id());
  for (auto& loss_tensor : train_loss_tensor_) {
    HCTR_LIB_THROW(cudaMemcpyAsync(&loss_temp[i], loss_tensor.second.data(), sizeof(float),
                                   cudaMemcpyDeviceToHost, gpu_resource_->get_stream()));
    ++i;
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(gpu_resource_->get_stream()));
  for (i = 0; i < train_loss_tensor_.size(); ++i) {
    loss_host += loss_temp[i];
  }
  delete loss_temp;
  return loss_host;
}

metrics::Core23MultiLossMetricMap Network::get_raw_metrics_all() const { return raw_metrics_; }

metrics::Core23RawMetricMap Network::get_raw_metrics(std::string loss_name) const {
  return raw_metrics_.find(loss_name)->second;
}

void Network::download_params_to_host(const std::string& write_path) {
  // forward
  CudaDeviceContext context(get_device_id());

  std::unique_ptr<char[]> weight(new char[train_weight_tensor_->num_bytes()]);
  HCTR_LIB_THROW(cudaMemcpy(weight.get(), train_weight_tensor_->data(),
                            train_weight_tensor_->num_bytes(), cudaMemcpyDeviceToHost));

  auto fs = FileSystemBuilder::build_unique_by_path(write_path);
  fs->write(write_path, weight.get(), train_weight_tensor_->num_bytes(), true);
  return;
}

void Network::download_opt_states_to_host(const std::string& write_path) {
  // forward
  CudaDeviceContext context(get_device_id());
  auto fs = FileSystemBuilder::build_unique_by_path(write_path);
  if (opt_tensor_->empty()) {
    fs->write(write_path, nullptr, 0, true);
    return;
  }
  size_t dst_size_in_byte = opt_tensor_->num_bytes();
  std::unique_ptr<char[]> h_opt_states(new char[dst_size_in_byte]);

  void* src = (void*)opt_tensor_->data();
  HCTR_LIB_THROW(cudaMemcpy(h_opt_states.get(), src, dst_size_in_byte, cudaMemcpyDeviceToHost));

  fs->write(write_path, h_opt_states.get(), dst_size_in_byte, true);
}

std::string Network::get_no_trained_params_in_string() {
  bool prev_exist = false;
  std::string net_str;
  for (auto& layer : train_layers_) {
    std::string layer_str = layer->get_no_trained_params_in_string();
    if (layer_str.length() != 0) {
      if (prev_exist) net_str += ",\n";
      net_str += "    {\n";
      net_str += layer_str;
      net_str += "\n    }";
      prev_exist = true;
    }
  }
  if (net_str.length() != 0) {
    net_str = "{\n  \"layers\": [\n" + net_str;
    net_str += "\n  ]\n}\n";
  }

  return net_str;
}

void Network::upload_params_to_device(const std::string& model_file) {
  auto fs = FileSystemBuilder::build_unique_by_path(model_file);
  CudaDeviceContext context(get_device_id());

  std::unique_ptr<char[]> params(new char[train_weight_tensor_->num_bytes()]);
  fs->read(model_file, params.get(), train_weight_tensor_->num_bytes(), 0);
  HCTR_LIB_THROW(cudaMemcpy(train_weight_tensor_->data(), params.get(),
                            train_weight_tensor_->num_bytes(), cudaMemcpyHostToDevice));
  return;
}

void Network::download_params_to_host(float* weight) {
  CudaDeviceContext context(get_device_id());

  HCTR_LIB_THROW(cudaMemcpy(weight, train_weight_tensor_->data(), train_weight_tensor_->num_bytes(),
                            cudaMemcpyDeviceToHost));

  return;
}

void Network::upload_params_to_device(float* params) {
  CudaDeviceContext context(get_device_id());

  HCTR_LIB_THROW(cudaMemcpy(train_weight_tensor_->data(), params, train_weight_tensor_->num_bytes(),
                            cudaMemcpyHostToDevice));

  return;
}

void Network::upload_opt_states_to_device(char* h_opt_states) {
  CudaDeviceContext context(get_device_id());

  size_t src_size_in_byte = opt_tensor_->num_bytes();

  void* dst = (void*)opt_tensor_->data();

  HCTR_LIB_THROW(cudaMemcpy(dst, h_opt_states, src_size_in_byte, cudaMemcpyHostToDevice));

  return;
}

void Network::init_params(size_t index) {
  CudaDeviceContext context(get_device_id());
  for (auto& layer : train_layers_) {
    layer->init_params(cpu_resource_->get_replica_uniform_curand_generator(index));
  }
}

void Network::exchange_wgrad() {
  CudaDeviceContext context(get_device_id());
  if (use_mixed_precision_) {
    HCTR_LIB_THROW(ncclAllReduce((const void*)wgrad_tensor_half_->data(),
                                 (void*)wgrad_tensor_half_->data(),
                                 wgrad_tensor_half_->num_elements(), ncclHalf, ncclSum,
                                 gpu_resource_->get_nccl(), gpu_resource_->get_stream()));
  } else {
    HCTR_LIB_THROW(ncclAllReduce((const void*)wgrad_tensor_->data(), (void*)wgrad_tensor_->data(),
                                 wgrad_tensor_->num_elements(), ncclFloat, ncclSum,
                                 gpu_resource_->get_nccl(), gpu_resource_->get_stream()));
  }
}

void Network::update_params() {
  optimizer_->update();
  return;
}

void Network::initialize(bool is_train) {
  CudaDeviceContext context(get_device_id());
  for (auto& layer : train_layers_) {
    layer->initialize();
  }
  for (auto& layer : evaluate_layers_) {
    layer->initialize();
  }
  if (is_train) {
    optimizer_->initialize();
  }
}

void Network::search_algorithm() {
  for (auto& layer : train_layers_) {
    layer->search_algorithm();
  }
  for (auto& layer : evaluate_layers_) {
    layer->search_algorithm();
  }
}

void Network::copy_weights_from_train_layers_to_evaluate_layers() {
  CudaDeviceContext context(get_device_id());
  HCTR_LIB_THROW(cudaMemcpyAsync(evaluate_weight_tensor_->data(), train_weight_tensor_->data(),
                                 train_weight_tensor_->num_bytes(), cudaMemcpyDeviceToDevice,
                                 gpu_resource_->get_stream()));

  if (use_mixed_precision_) {
    conv_weight_(evaluate_weight_tensor_half_, evaluate_weight_tensor_);
  }
}

void Network::copy_non_trainable_params_from_train_layers_to_evaluate_layers() {
  CudaDeviceContext context(get_device_id());
  for (size_t i{0}; i < train_layers_.size(); ++i) {
    auto tensors_in_train_layers = train_layers_[i]->get_non_trainable_params_as_tensors();
    auto tensors_in_evaluate_layers = evaluate_layers_[i]->get_non_trainable_params_as_tensors();
    for (size_t j{0}; j < tensors_in_train_layers.size(); ++j) {
      HCTR_LIB_THROW(cudaMemcpyAsync(tensors_in_evaluate_layers[j].data(),
                                     tensors_in_train_layers[j].data(),
                                     tensors_in_train_layers[j].num_bytes(),
                                     cudaMemcpyDeviceToDevice, gpu_resource_->get_stream()));
    }
  }
}

void Network::set_train_layers(std::vector<std::unique_ptr<Layer>>&& train_layers) {
  if (use_mixed_precision_) {
    train_weight_tensor_half_ = get_trainable_tensors<__half, __half>(
        train_layers, [](auto& layer) -> auto{ return layer->get_weights(); });
    train_weight_tensor_ = get_trainable_tensors<__half, float>(
        train_layers, [](auto& layer) -> auto{ return layer->get_master_weights(); });
    wgrad_tensor_half_ = get_trainable_tensors<__half, __half>(
        train_layers, [](auto& layer) -> auto{ return layer->get_wgrads(); });
  } else {
    train_weight_tensor_ = get_trainable_tensors<float, float>(
        train_layers, [](auto& layer) -> auto{ return layer->get_weights(); });
    wgrad_tensor_ = get_trainable_tensors<float, float>(
        train_layers, [](auto& layer) -> auto{ return layer->get_wgrads(); });
  }
  train_layers_ = std::move(train_layers);
}
void Network::set_evaluate_layers(std::vector<std::unique_ptr<Layer>>&& evaluate_layers) {
  if (use_mixed_precision_) {
    evaluate_weight_tensor_half_ = get_trainable_tensors<__half, __half>(
        evaluate_layers, [](auto& layer) -> auto{ return layer->get_weights(); });
    evaluate_weight_tensor_ = get_trainable_tensors<__half, float>(
        evaluate_layers, [](auto& layer) -> auto{ return layer->get_master_weights(); });
  } else {
    evaluate_weight_tensor_ = get_trainable_tensors<float, float>(
        evaluate_layers, [](auto& layer) -> auto{ return layer->get_weights(); });
  }
  evaluate_layers_ = std::move(evaluate_layers);
}

void Network::set_train_losses(std::map<std::string, std::unique_ptr<ILoss>>&& train_losses,
                               const std::map<std::string, float>& label_weights) {
  set_losses_common(train_losses, label_weights, train_loss_tensor_);
  train_losses_ = std::move(train_losses);
}
void Network::set_evaluate_losses(std::map<std::string, std::unique_ptr<ILoss>>&& evaluate_losses,
                                  const std::map<std::string, float>& label_weights) {
  set_losses_common(evaluate_losses, label_weights, evaluate_loss_tensor_);
  evaluate_losses_ = std::move(evaluate_losses);
}

void Network::set_top_and_bottom_layers(std::vector<Layer*>&& top_layers,
                                        std::vector<Layer*>&& bottom_layers) {
  top_layers_ = std::move(top_layers);
  bottom_layers_ = std::move(bottom_layers);
}

void Network::set_raw_metrics(metrics::Core23MultiLossMetricMap&& raw_metrics) {
  raw_metrics_ = std::move(raw_metrics);
}

void Network::set_optimizer(std::unique_ptr<Optimizer> optimizer) {
  auto opt_tensors = optimizer->get_opt_state_tensors();
  int64_t num_opt_tensors = opt_tensors.size();
  opt_tensor_.emplace(opt_tensors, core23::Shape({num_opt_tensors}));
  optimizer_ = std::move(optimizer);
}

void Network::create_and_set_optimizer(const OptParams& opt_params) {
  if (use_mixed_precision_) {
    auto weight_tensors = get_master_weight_tensor_vector<__half>(train_layers_);
    auto weight_half_tensors = get_weight_tensor_vector<__half>(train_layers_);
    auto wgrad_tensors = get_wgrad_tensor_vector<__half>(train_layers_);
    optimizer_ =
        Optimizer::Create<__half>(opt_params, weight_tensors, weight_half_tensors, wgrad_tensors,
                                  opt_params.scaler, gpu_resource_, use_mixed_precision_);
  } else {
    auto weight_tensors = get_weight_tensor_vector<float>(train_layers_);
    auto weight_half_tensors = std::vector<core23::Tensor>();
    auto wgrad_tensors = get_wgrad_tensor_vector<float>(train_layers_);
    optimizer_ =
        Optimizer::Create<float>(opt_params, weight_tensors, weight_half_tensors, wgrad_tensors,
                                 opt_params.scaler, gpu_resource_, use_mixed_precision_);
  }
  auto opt_tensors = optimizer_->get_opt_state_tensors();
  int64_t num_opt_tensors = opt_tensors.size();
  opt_tensor_.emplace(opt_tensors, core23::Shape({num_opt_tensors}));
}

void Network::conv_weight_(std::optional<core23::TensorContainer<__half, 1, 1>>& target_opt,
                           const std::optional<core23::TensorContainer<float, 1, 1>>& source_opt) {
  CudaDeviceContext context(get_device_id());
  auto& target = target_opt.value();
  const auto& source = source_opt.value();
  size_t elems = source.num_elements();
  if (target.num_elements() != source.num_elements())
    HCTR_OWN_THROW(Error_t::WrongInput, "weight size of target != weight size of in");
  const size_t BLOCK = 256;
  size_t GRID = (elems - 1) / BLOCK + 1;
  GRID = GRID > 10 * gpu_resource_->get_sm_count() ? 10 * gpu_resource_->get_sm_count() : GRID;
  conv_weight_gpu(GRID, BLOCK, target.flatten().data(), source.flatten().data(), elems,
                  gpu_resource_->get_stream());
}

void Network::prop_layers(const std::vector<Layer*>& layers, bool fprop, bool train) {
  if (fprop) {
    for (auto& layer : layers) {
      layer->fprop(train);
    }
  } else {
    for (auto it = layers.rbegin(); it != layers.rend(); it++) {
      (*it)->bprop();
    }
  }
}

void Network::set_losses_common(const std::map<std::string, std::unique_ptr<ILoss>>& losses,
                                const std::map<std::string, float>& label_weights,
                                std::map<std::string, core23::Tensor>& loss_tensors) {
  for (auto& pair : losses) {
    if (use_mixed_precision_) {
      auto loss_ptr = dynamic_cast<Loss<__half>*>(pair.second.get());
      loss_tensors[pair.first] = loss_ptr->get_loss_tensors()[0];
    } else {
      auto loss_ptr = dynamic_cast<Loss<float>*>(pair.second.get());
      loss_tensors[pair.first] = loss_ptr->get_loss_tensors()[0];
    }
    auto it = label_weights.find(pair.first);
    if (it == label_weights.end()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Label weight names and losses do not match.");
    }
    pair.second->set_label_weight(it->second);
  }
}

}  // namespace HugeCTR

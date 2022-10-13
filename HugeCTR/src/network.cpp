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

#include <omp.h>

#include <io/filesystem.hpp>
#include <layers/fully_connected_layer.hpp>
#include <layers/relu_layer.hpp>
#include <network.hpp>
#include <parser.hpp>
#include <regularizers/no_regularizer.hpp>

namespace HugeCTR {

void conv_weight_gpu(size_t grid, size_t block, __half* dst, const float* src, int elems,
                     cudaStream_t stream);

Network::Network(const std::shared_ptr<CPUResource>& cpu_resource,
                 const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision)
    : cpu_resource_(cpu_resource),
      gpu_resource_(gpu_resource),
      use_mixed_precision_(use_mixed_precision) {}

void Network::update_params() {
  optimizer_->update();
  return;
}

void Network::conv_weight_(Tensor2<__half>& target, const Tensor2<float>& source) {
  CudaDeviceContext context(get_device_id());
  size_t elems = source.get_num_elements();
  if (target.get_num_elements() != source.get_num_elements())
    HCTR_OWN_THROW(Error_t::WrongInput, "weight size of target != weight size of in");
  const size_t BLOCK = 256;
  size_t GRID = (elems - 1) / BLOCK + 1;
  GRID = GRID > 10 * gpu_resource_->get_sm_count() ? 10 * gpu_resource_->get_sm_count() : GRID;
  conv_weight_gpu(GRID, BLOCK, target.get_ptr(), source.get_ptr(), elems,
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

// TODO - Update this method for multi-regularzer to run each regularizer and compute the
// conditional weighted sum for each layer.  Then similarly compute the weighted sum of losses
// associated with each layer
void Network::train(long long current_batchsize) {
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

void Network::eval(long long current_batchsize) {
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

void Network::download_params_to_host(std::string& write_path) {
  // forward
  CudaDeviceContext context(get_device_id());

  std::unique_ptr<char[]> weight(new char[train_weight_tensor_.get_size_in_bytes()]);
  HCTR_LIB_THROW(cudaMemcpy(weight.get(), train_weight_tensor_.get_ptr(),
                            train_weight_tensor_.get_size_in_bytes(), cudaMemcpyDeviceToHost));

  auto fs = FileSystemBuilder::build_unique_by_path(write_path);
  fs->write(write_path, weight.get(), train_weight_tensor_.get_size_in_bytes(), true);
  return;
}

void Network::download_opt_states_to_host(std::string& write_path) {
  // forward
  CudaDeviceContext context(get_device_id());

  size_t dst_size_in_byte =
      use_mixed_precision_ ? opt_tensor_half_.get_size_in_bytes() : opt_tensor_.get_size_in_bytes();
  std::unique_ptr<char[]> h_opt_states(new char[dst_size_in_byte]);

  void* src =
      use_mixed_precision_ ? (void*)opt_tensor_half_.get_ptr() : (void*)opt_tensor_.get_ptr();
  HCTR_LIB_THROW(cudaMemcpy(h_opt_states.get(), src, dst_size_in_byte, cudaMemcpyDeviceToHost));

  auto fs = FileSystemBuilder::build_unique_by_path(write_path);
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
  std::ifstream model_stream(model_file, std::ifstream::binary);
  if (!model_stream.is_open()) {
    HCTR_OWN_THROW(Error_t::WrongInput, std::string("Cannot open dense model file (reason: ") +
                                            std::strerror(errno) + ")");
  }
  CudaDeviceContext context(get_device_id());

  std::unique_ptr<char[]> params(new char[train_weight_tensor_.get_size_in_bytes()]);
  model_stream.read(params.get(), train_weight_tensor_.get_size_in_bytes());
  HCTR_LIB_THROW(cudaMemcpy(train_weight_tensor_.get_ptr(), params.get(),
                            train_weight_tensor_.get_size_in_bytes(), cudaMemcpyHostToDevice));
  model_stream.close();
  return;
}

void Network::upload_params_to_device_inference(const std::string& model_file) {
  std::ifstream model_stream(model_file, std::ifstream::binary);
  if (!model_stream.is_open()) {
    HCTR_OWN_THROW(Error_t::WrongInput, std::string("Cannot open dense model file (reason: ") +
                                            std::strerror(errno) + ")");
  }
  CudaDeviceContext context(get_device_id());

  std::unique_ptr<char[]> params(new char[evaluate_weight_tensor_.get_size_in_bytes()]);
  model_stream.read(params.get(), evaluate_weight_tensor_.get_size_in_bytes());
  HCTR_LIB_THROW(cudaMemcpyAsync(evaluate_weight_tensor_.get_ptr(), params.get(),
                                 evaluate_weight_tensor_.get_size_in_bytes(),
                                 cudaMemcpyHostToDevice, gpu_resource_->get_stream()));
  model_stream.close();
  if (use_mixed_precision_) {
    conv_weight_(evaluate_weight_tensor_half_, evaluate_weight_tensor_);
  }
  return;
}

void Network::upload_non_trainable_params_to_device_inference(const std::string& model_file) {
  HCTR_LOG(INFO, ROOT, "Upload non-trainable parameters from JSON file to inference layers\n");
  const nlohmann::json& params_json(read_json_file(model_file));
  const nlohmann::json& params_for_layers = get_json(params_json, "layers");
  size_t counter = 0;
  CudaDeviceContext context(get_device_id());
  for (size_t i{0}; i < evaluate_layers_.size(); ++i) {
    auto params_tensors = evaluate_layers_[i]->get_tensors_for_non_trainable_params();
    if (params_tensors.size() > 1) {
      const nlohmann::json& params = params_for_layers[counter];
      std::string layer_type = get_value_from_json<std::string>(params, "type");
      if (layer_type == "BatchNorm") {
        std::vector<float> running_mean = get_json(params, "mean");
        std::vector<float> running_variance = get_json(params, "var");
        HCTR_LIB_THROW(cudaMemcpyAsync(params_tensors[0].get_ptr(), running_mean.data(),
                                       params_tensors[0].get_size_in_bytes(),
                                       cudaMemcpyHostToDevice, gpu_resource_->get_stream()));
        HCTR_LIB_THROW(cudaMemcpyAsync(params_tensors[1].get_ptr(), running_variance.data(),
                                       params_tensors[1].get_size_in_bytes(),
                                       cudaMemcpyHostToDevice, gpu_resource_->get_stream()));
      } else {
        HCTR_OWN_THROW(Error_t::WrongInput, "Only BatchNorm layer has non-trainable parameters");
      }
      ++counter;
    }
  }
}

void Network::download_params_to_host(float* weight) {
  CudaDeviceContext context(get_device_id());

  HCTR_LIB_THROW(cudaMemcpy(weight, train_weight_tensor_.get_ptr(),
                            train_weight_tensor_.get_size_in_bytes(), cudaMemcpyDeviceToHost));

  return;
}

void Network::upload_params_to_device(float* params) {
  CudaDeviceContext context(get_device_id());

  HCTR_LIB_THROW(cudaMemcpy(train_weight_tensor_.get_ptr(), params,
                            train_weight_tensor_.get_size_in_bytes(), cudaMemcpyHostToDevice));

  return;
}

void Network::upload_opt_states_to_device(char* h_opt_states) {
  CudaDeviceContext context(get_device_id());

  size_t src_size_in_byte =
      use_mixed_precision_ ? opt_tensor_half_.get_size_in_bytes() : opt_tensor_.get_size_in_bytes();

  void* dst =
      use_mixed_precision_ ? (void*)opt_tensor_half_.get_ptr() : (void*)opt_tensor_.get_ptr();

  HCTR_LIB_THROW(cudaMemcpy(dst, h_opt_states, src_size_in_byte, cudaMemcpyHostToDevice));

  return;
}

void Network::init_params(size_t index) {
  CudaDeviceContext context(get_device_id());
  for (auto& layer : train_layers_) {
    layer->init_params(cpu_resource_->get_replica_uniform_curand_generator(index));
  }
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

float Network::get_loss() {
  float loss_host = 0.f;
  float* loss_temp = new float[train_loss_tensors_.size()];
  size_t i = 0;

  CudaDeviceContext context(get_device_id());
  for (auto& loss_tensor : train_loss_tensors_) {
    HCTR_LIB_THROW(cudaMemcpyAsync(&loss_temp[i], loss_tensor.second.get_ptr(), sizeof(float),
                                   cudaMemcpyDeviceToHost, gpu_resource_->get_stream()));
    ++i;
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(gpu_resource_->get_stream()));
  for (i = 0; i < train_loss_tensors_.size(); ++i) {
    loss_host += loss_temp[i];
  }
  delete loss_temp;
  return loss_host;
}

std::map<std::string, metrics::RawMetricMap> Network::get_raw_metrics_all() const {
  return raw_metrics_;
}

metrics::RawMetricMap Network::get_raw_metrics(std::string loss_name) const {
  return raw_metrics_.find(loss_name)->second;
}

void Network::exchange_wgrad() {
  CudaDeviceContext context(get_device_id());
  if (use_mixed_precision_) {
    HCTR_LIB_THROW(ncclAllReduce((const void*)wgrad_tensor_half_.get_ptr(),
                                 (void*)wgrad_tensor_half_.get_ptr(),
                                 wgrad_tensor_half_.get_num_elements(), ncclHalf, ncclSum,
                                 gpu_resource_->get_nccl(), gpu_resource_->get_stream()));
  } else {
    HCTR_LIB_THROW(ncclAllReduce((const void*)wgrad_tensor_.get_ptr(),
                                 (void*)wgrad_tensor_.get_ptr(), wgrad_tensor_.get_num_elements(),
                                 ncclFloat, ncclSum, gpu_resource_->get_nccl(),
                                 gpu_resource_->get_stream()));
  }
}

void Network::copy_weights_from_train_layers_to_evaluate_layers() {
  // HCTR_LOG(INFO, ROOT, "Copying trainable weights from train layers to evaluate layers\n");
  CudaDeviceContext context(get_device_id());
  HCTR_LIB_THROW(cudaMemcpyAsync(evaluate_weight_tensor_.get_ptr(), train_weight_tensor_.get_ptr(),
                                 train_weight_tensor_.get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                 gpu_resource_->get_stream()));

  if (use_mixed_precision_) {
    conv_weight_(evaluate_weight_tensor_half_, evaluate_weight_tensor_);
  }
}

void Network::copy_non_trainable_params_from_train_layers_to_evaluate_layers() {
  // HCTR_LOG(INFO, ROOT, "Copying non-trainable parameters from train layers to evaluate
  // layers\n");
  CudaDeviceContext context(get_device_id());
  for (size_t i{0}; i < train_layers_.size(); ++i) {
    auto tensors_in_train_layers = train_layers_[i]->get_tensors_for_non_trainable_params();
    auto tensors_in_evaluate_layers = evaluate_layers_[i]->get_tensors_for_non_trainable_params();
    for (size_t j{0}; j < tensors_in_train_layers.size(); ++j) {
      HCTR_LIB_THROW(cudaMemcpyAsync(tensors_in_evaluate_layers[j].get_ptr(),
                                     tensors_in_train_layers[j].get_ptr(),
                                     tensors_in_train_layers[j].get_size_in_bytes(),
                                     cudaMemcpyDeviceToDevice, gpu_resource_->get_stream()));
    }
  }
}

}  // namespace HugeCTR

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

#include "HugeCTR/include/network.hpp"
#include "HugeCTR/include/layers/fully_connected_layer.hpp"
#include "HugeCTR/include/layers/relu_layer.hpp"
#include "HugeCTR/include/optimizers/momentum_sgd.hpp"
#include "HugeCTR/include/regularizers/no_regularizer.hpp"

namespace HugeCTR {


Network::Network(int batch_size, int device_id, const std::shared_ptr<const GPUResource>& gpu_resource,
                 bool disable_parser)
    : blobs_buff_(new GeneralBuffer<float>()),
      weight_buff_(new GeneralBuffer<float>()),
      wgrad_buff_(new GeneralBuffer<float>()),
      gpu_resource_(gpu_resource),
      device_id_(device_id),
      batch_size_(batch_size){
  return;
}

void Network::update_params() {
  optimizer_->update(gpu_resource_->get_stream());
  return;
}

void Network::train() {
#ifndef NDEBUG
  print_buffer(*weight_buff_, 18, 38);
  print_buffer(*weight_buff_, -20, -1);
  print_buffer(*wgrad_buff_, 18, 38);
  print_buffer(*wgrad_buff_, -20, -1);

  for (auto& tensor : tensors_) {
    print_tensor(*tensor, -10, -1);
  }
#endif
  // forward
  for (auto& layer : layers_) {
    layer->fprop(gpu_resource_->get_stream());
  }
  loss_->fused_loss_computation(gpu_resource_->get_stream());
  // backward
  for (auto it = layers_.rbegin(); it != layers_.rend(); it++) {
    (*it)->bprop(gpu_resource_->get_stream());
  }
  return;
}

void Network::eval() {
#ifndef NDEBUG
  print_buffer(*weight_buff_, 18, 38);
  print_buffer(*weight_buff_, -20, -1);
  print_buffer(*wgrad_buff_, 18, 38);
  print_buffer(*wgrad_buff_, -20, -1);

  for (auto& tensor : tensors_) {
    print_tensor(*tensor, -10, -1);
  }
#endif
  // forward
  for (auto& layer : layers_) {
    //layer->fprop(gpu_resource_->get_stream());
    layer->inference(gpu_resource_->get_stream());
#ifndef NDEBUG
    for (auto& tensor : tensors_) {
      print_tensor(*tensor, -10, -1);
    }
#endif
  }
  loss_->fused_loss_computation(gpu_resource_->get_stream());
#ifndef NDEBUG
  for (auto& tensor : tensors_) {
    print_tensor(*tensor, -10, -1);
  }
#endif

  return;
}

void Network::download_params_to_host(std::ofstream& weight_stream) {
  // forward
  CudaDeviceContext context(device_id_);

  std::unique_ptr<char[]> weight(new char[weight_buff_->get_size()]);
  CK_CUDA_THROW_(cudaMemcpy(weight.get(), weight_buff_->get_ptr_with_offset(0),
                            weight_buff_->get_size(), cudaMemcpyDeviceToHost));
  weight_stream.write(weight.get(), weight_buff_->get_size());

  return;
}

std::string Network::get_no_trained_params_in_string() {
  bool prev_exist = false;
  std::string net_str;
  for (auto& layer : layers_) {
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
    CK_THROW_(Error_t::WrongInput, "Cannot open dense model file");
  }
  CudaDeviceContext context(device_id_);

  std::unique_ptr<char[]> params(new char[weight_buff_->get_size()]);
  model_stream.read(params.get(), weight_buff_->get_size());
  CK_CUDA_THROW_(cudaMemcpy(weight_buff_->get_ptr_with_offset(0), params.get(),
                            weight_buff_->get_size(), cudaMemcpyHostToDevice));
  model_stream.close();
  return;
}

void Network::download_params_to_host(float* weight) {
  CudaDeviceContext context(device_id_);

  CK_CUDA_THROW_(cudaMemcpy(weight, weight_buff_->get_ptr_with_offset(0), weight_buff_->get_size(),
                            cudaMemcpyDeviceToHost));

  return;
}

void Network::upload_params_to_device(float* params) {
  CudaDeviceContext context(device_id_);

  CK_CUDA_THROW_(cudaMemcpy(weight_buff_->get_ptr_with_offset(0), params, weight_buff_->get_size(),
                            cudaMemcpyHostToDevice));

  return;
}

// void Network::init_params(std::ofstream& out_stream) {
//   for (auto& layer : layers_) layer->init_params(out_stream);
// }

void Network::init_params(const std::string& dense_name) {
  std::ofstream out_stream(dense_name, std::ofstream::binary);
  if (!out_stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Cannot open dense model file");
  }
  for (auto& layer : layers_) layer->init_params(out_stream);
  out_stream.close();
}

void Network::copy_params(const Network& n) {
  assert(weight_buff_->get_size() == n.weight_buff_->get_size());
  CK_CUDA_THROW_(cudaMemcpy(weight_buff_->get_ptr_with_offset(0), n.weight_buff_->get_ptr_with_offset(0), weight_buff_->get_size(),cudaMemcpyDeviceToDevice));
}


float Network::get_loss() {
  float loss_host = 0.f;

  CudaDeviceContext context(device_id_);

  CK_CUDA_THROW_(
      cudaMemcpy(&loss_host, loss_tensor_->get_ptr(), sizeof(float), cudaMemcpyDeviceToHost));

  return loss_host;
}

void Network::exchange_wgrad() {
  if (gpu_resource_->get_nccl_ptr() != nullptr) {
    CudaDeviceContext context(device_id_);

    CK_NCCL_THROW_(ncclAllReduce((const void*)wgrad_buff_->get_ptr_with_offset(0),
                                 (void*)wgrad_buff_->get_ptr_with_offset(0),
                                 wgrad_buff_->get_num_elements(), ncclFloat, ncclSum,
                                 *(gpu_resource_->get_nccl_ptr()), gpu_resource_->get_stream()));
  } else {
    CK_THROW_(Error_t::IllegalCall, "cannot call exchange_wgrad with single GPU");
  }
}

}  // namespace HugeCTR

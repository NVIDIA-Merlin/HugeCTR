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

namespace HugeCTR {

Network::Network(Tensor<float>& in_tensor, const Tensor<float>& label_tensor, int batchsize,
                 int device_id, const GPUResource* gpu_resource, bool disable_parser)
    : gpu_resource_(*gpu_resource),
      device_id_(device_id),
      batchsize_(batchsize),
      in_tensor_(in_tensor),
      label_tensor_(label_tensor) {
  if (disable_parser) {
    try {
      std::vector<int> tmp_dim;

      /* setup network */
      assert(tensors_.empty());
      assert(layers_.empty());

      // FC 0 xxx->200
      tensors_.push_back(
          new Tensor<float>(tmp_dim = {batchsize, 200}, blobs_buff_, TensorFormat_t::HW));
      layers_.push_back(new FullyConnectedLayer(weight_buff_, wgrad_buff_, (in_tensor_),
                                                *tensors_[0], TensorFormat_t::HW,
                                                *gpu_resource_.get_cublas_handle_ptr(), device_id));
      tensors_.push_back(
          new Tensor<float>(tmp_dim = {batchsize, 200}, blobs_buff_, TensorFormat_t::HW));
      layers_.push_back(new ReluLayer(*tensors_[0], *tensors_[1], device_id));
      // FC 1 200->200
      tensors_.push_back(
          new Tensor<float>(tmp_dim = {batchsize, 200}, blobs_buff_, TensorFormat_t::HW));
      layers_.push_back(new FullyConnectedLayer(weight_buff_, wgrad_buff_, *tensors_[1],
                                                *tensors_[2], TensorFormat_t::HW,
                                                *gpu_resource_.get_cublas_handle_ptr(), device_id));
      tensors_.push_back(
          new Tensor<float>(tmp_dim = {batchsize, 200}, blobs_buff_, TensorFormat_t::HW));
      layers_.push_back(new ReluLayer(*tensors_[2], *tensors_[3], device_id));
      // FC 2 200->200
      tensors_.push_back(
          new Tensor<float>(tmp_dim = {batchsize, 200}, blobs_buff_, TensorFormat_t::HW));
      layers_.push_back(new FullyConnectedLayer(weight_buff_, wgrad_buff_, *tensors_[3],
                                                *tensors_[4], TensorFormat_t::HW,
                                                *gpu_resource_.get_cublas_handle_ptr(), device_id));
      tensors_.push_back(
          new Tensor<float>(tmp_dim = {batchsize, 200}, blobs_buff_, TensorFormat_t::HW));
      layers_.push_back(new ReluLayer(*tensors_[4], *tensors_[5], device_id));
      // FC 3 200->1
      tensors_.push_back(
          new Tensor<float>(tmp_dim = {batchsize, 1}, blobs_buff_, TensorFormat_t::HW));
      layers_.push_back(new FullyConnectedLayer(weight_buff_, wgrad_buff_, *tensors_[5],
                                                *tensors_[6], TensorFormat_t::HW,
                                                *gpu_resource_.get_cublas_handle_ptr(), device_id));
      // setup loss
      loss_tensor_ = new Tensor<float>(tmp_dim = {1, 1}, blobs_buff_, TensorFormat_t::HW);
      loss_ = new BinaryCrossEntropyLoss(const_cast<Tensor<float>&>(label_tensor_),
                                         *tensors_.back(), *loss_tensor_, device_id);

      // setup optimizer
      optimizer_ = new MomentumSGD(weight_buff_, wgrad_buff_, device_id, 0.01, 0.9);

      weight_buff_.init(device_id);
      wgrad_buff_.init(device_id);
      blobs_buff_.init(device_id);

    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }
  return;
}

void Network::update_params() {
  optimizer_->update(*gpu_resource_.get_stream_ptr());
  return;
}

void Network::train() {
#ifndef NDEBUG
  print_buffer(weight_buff_, 18, 38);
  print_buffer(weight_buff_, -20, -1);
  print_buffer(wgrad_buff_, 18, 38);
  print_buffer(wgrad_buff_, -20, -1);

  print_tensor(in_tensor_, -10, -1);
  print_tensor(label_tensor_, -10, -1);
  for (auto tensor : tensors_) {
    print_tensor(*tensor, -10, -1);
  }
#endif
  // forward
  for (auto iter = layers_.begin(); iter != layers_.end(); iter++) {
    iter[0]->fprop(*gpu_resource_.get_stream_ptr());
#ifndef NDEBUG
    print_tensor(in_tensor_, -10, -1);
    print_tensor(label_tensor_, -10, -1);
    for (auto tensor : tensors_) {
      print_tensor(*tensor, -10, -1);
    }
#endif
  }
  loss_->fused_loss_computation(*gpu_resource_.get_stream_ptr());
#ifndef NDEBUG
  print_tensor(in_tensor_, -10, -1);
  print_tensor(label_tensor_, -10, -1);
  for (auto tensor : tensors_) {
    print_tensor(*tensor, -10, -1);
  }
#endif

  // backward
  for (auto iter = layers_.rbegin(); iter != layers_.rend(); iter++) {
    iter[0]->bprop(*gpu_resource_.get_stream_ptr());
#ifndef NDEBUG
    print_tensor(in_tensor_, -10, -1);
    print_tensor(label_tensor_, -10, -1);
    for (auto tensor : tensors_) {
      print_tensor(*tensor, -10, -1);
    }
#endif
  }
  return;
}

void Network::eval() {
#ifndef NDEBUG
  print_buffer(weight_buff_, 18, 38);
  print_buffer(weight_buff_, -20, -1);
  print_buffer(wgrad_buff_, 18, 38);
  print_buffer(wgrad_buff_, -20, -1);

  print_tensor(in_tensor_, -10, -1);
  print_tensor(label_tensor_, -10, -1);
  for (auto tensor : tensors_) {
    print_tensor(*tensor, -10, -1);
  }
#endif
  // forward
  for (auto iter = layers_.begin(); iter != layers_.end(); iter++) {
    iter[0]->fprop(*gpu_resource_.get_stream_ptr());
#ifndef NDEBUG
    print_tensor(in_tensor_, -10, -1);
    print_tensor(label_tensor_, -10, -1);
    for (auto tensor : tensors_) {
      print_tensor(*tensor, -10, -1);
    }
#endif
  }
  loss_->fused_loss_computation(*gpu_resource_.get_stream_ptr());
#ifndef NDEBUG
  print_tensor(in_tensor_, -10, -1);
  print_tensor(label_tensor_, -10, -1);
  for (auto tensor : tensors_) {
    print_tensor(*tensor, -10, -1);
  }
#endif

  return;
}

void Network::download_params_to_host(std::ofstream& weight_stream) {
  // forward
  CudaDeviceContext context(device_id_);

  float* weight = (float*)malloc(weight_buff_.get_size());
  CK_CUDA_THROW_(cudaMemcpy(weight, weight_buff_.get_ptr_with_offset(0), weight_buff_.get_size(),
                            cudaMemcpyDeviceToHost));
  weight_stream.write(reinterpret_cast<char*>(weight), weight_buff_.get_size());
  free(weight);

  return;
}

std::string Network::get_no_trained_params_in_string() {
  bool prev_exist = false;
  std::string net_str;
  for (auto layer : layers_) {
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

void Network::upload_params_to_device(std::ifstream& params_stream) {
  CudaDeviceContext context(device_id_);

  float* params = (float*)malloc(weight_buff_.get_size());
  params_stream.read(reinterpret_cast<char*>(params), weight_buff_.get_size());
  CK_CUDA_THROW_(cudaMemcpy(weight_buff_.get_ptr_with_offset(0), params, weight_buff_.get_size(),
                            cudaMemcpyHostToDevice));

  return;
}

void Network::download_params_to_host(float* weight) {
  CudaDeviceContext context(device_id_);

  CK_CUDA_THROW_(cudaMemcpy(weight, weight_buff_.get_ptr_with_offset(0), weight_buff_.get_size(),
                            cudaMemcpyDeviceToHost));

  return;
}

void Network::upload_params_to_device(float* params) {
  CudaDeviceContext context(device_id_);

  CK_CUDA_THROW_(cudaMemcpy(weight_buff_.get_ptr_with_offset(0), params, weight_buff_.get_size(),
                            cudaMemcpyHostToDevice));

  return;
}

void Network::init_params(std::ofstream& out_stream) {
  for (auto layer : layers_) layer->init_params(out_stream);
}

float Network::get_loss() {
  float loss_host = 0.f;

  CudaDeviceContext context(device_id_);

  CK_CUDA_THROW_(
      cudaMemcpy(&loss_host, loss_tensor_->get_ptr(), sizeof(float), cudaMemcpyDeviceToHost));

  return loss_host;
}

void Network::exchange_wgrad() {
  if (gpu_resource_.get_nccl_ptr() != nullptr) {
    CudaDeviceContext context(device_id_);

    CK_NCCL_THROW_(ncclAllReduce(
        (const void*)wgrad_buff_.get_ptr_with_offset(0), (void*)wgrad_buff_.get_ptr_with_offset(0),
        wgrad_buff_.get_num_elements(), ncclFloat, ncclSum, *(gpu_resource_.get_nccl_ptr()),
        *(gpu_resource_.get_stream_ptr())));
  } else {
    CK_THROW_(Error_t::IllegalCall, "cannot call exchange_wgrad with single GPU");
  }
}

Network::~Network() {
  try {
    assert(optimizer_ != nullptr && loss_ != nullptr && loss_tensor_ != nullptr);
    delete optimizer_;
    delete loss_;
    delete loss_tensor_;
    for (auto tensor : tensors_) {
      assert(tensor != nullptr);
      delete tensor;
    }

    for (auto layer : layers_) {
      assert(layer != nullptr);
      delete layer;
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

}  // namespace HugeCTR

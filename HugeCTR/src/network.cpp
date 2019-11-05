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

void Network::train(int local_gpu_count) {
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

  int fp_loop = 0;
  // forward
  for (auto iter = layers_.begin(); iter != layers_.end(); iter++) {
    iter[0]->fprop(*gpu_resource_.get_stream_ptr());
    if (fp_loop == 1 && is_speedup_ == 1) {  // get fc1_output_tensor
      auto fc_output_tensor = tensors_[1];
      // use NCCL to do Reduce-Scatter
      int batchsize_per_gpu = (int)(fc_output_tensor->get_dims()[0] / local_gpu_count);
      int recv_count = batchsize_per_gpu * fc_output_tensor->get_dims()[1];
      if(local_gpu_count > 1) { // local_gpu_count > 1
        if(gpu_resource_.get_nccl_ptr() != nullptr){
          int old_device = -1;
          CK_CUDA_THROW_(get_set_device(device_id_, &old_device));
          Tensor<float>* output_tensor = forward_temp_tensors_;

          CK_NCCL_THROW_(ncclReduceScatter((const void*)fc_output_tensor->get_ptr(), // send buf
          (void*)output_tensor->get_ptr(), // recv buff
          recv_count, 
          ncclFloat, 
          ncclSum, 
          *(gpu_resource_.get_nccl_ptr()), *(gpu_resource_.get_stream_ptr())));
          CK_CUDA_THROW_(get_set_device(old_device));
        }
        else{
          CK_THROW_(Error_t::IllegalCall, "cannot call exchange_wgrad with single GPU");
        }

      } 
      else { // local_gpu_count == 1
        int old_device = -1;
        CK_CUDA_THROW_(get_set_device(device_id_, &old_device));
        Tensor<float>* output_tensor = forward_temp_tensors_;
        CK_CUDA_THROW_(cudaMemcpyAsync(output_tensor->get_ptr(), \
            fc_output_tensor->get_ptr(), \
            recv_count * sizeof(float), cudaMemcpyDeviceToDevice, \
            *(gpu_resource_.get_stream_ptr())));
        CK_CUDA_THROW_(get_set_device(old_device));
        CK_CUDA_THROW_(cudaStreamSynchronize(*(gpu_resource_.get_stream_ptr())));
      }
    }
    fp_loop++;

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

  int bp_loop = 0;
  // backward
  for (auto iter = layers_.rbegin(); iter != layers_.rend(); iter++) {
    iter[0]->bprop(*gpu_resource_.get_stream_ptr());
    if (bp_loop == (layers_.size() - 3) && is_speedup_ == 1) { // relu1_layer
      auto relu_in_tensors = iter[0]->get_in_tensor();
      Tensor<float> relu_in_tensor = relu_in_tensors[0];
      int batchsize_per_gpu = relu_in_tensor.get_dims()[0];
      // use NCCL to do All-Gather
      // each recv buffer(backward_temp_tensors_ = fc1_out_tensor) has the same top_grad data
      int send_count = batchsize_per_gpu * relu_in_tensor.get_dims()[1];
      if(local_gpu_count > 1) {
        if(gpu_resource_.get_nccl_ptr() != nullptr){
          int old_device1 = -1;
          CK_CUDA_THROW_(get_set_device(device_id_, &old_device1));

          Tensor<float>* output_tensor = backward_temp_tensors_;
          CK_NCCL_THROW_(ncclAllGather((const void*)relu_in_tensor.get_ptr(), // send buf
          (void*)output_tensor->get_ptr(), // recv buff
          send_count, 
          ncclFloat, 
          *(gpu_resource_.get_nccl_ptr()), *(gpu_resource_.get_stream_ptr())));
          CK_CUDA_THROW_(get_set_device(old_device1));
        }
        else{
          CK_THROW_(Error_t::IllegalCall, "cannot call exchange_wgrad with single GPU");
        }
      }
      else { // local_gpu_count == 1
        int old_device1 = -1;
        CK_CUDA_THROW_(get_set_device(device_id_, &old_device1));
        Tensor<float>* output_tensor = backward_temp_tensors_;
        CK_CUDA_THROW_(cudaMemcpyAsync(output_tensor->get_ptr(), \
            relu_in_tensor.get_ptr(), send_count * sizeof(float), cudaMemcpyDeviceToDevice, \
            *(gpu_resource_.get_stream_ptr())));
        CK_CUDA_THROW_(get_set_device(old_device1));
        CK_CUDA_THROW_(cudaStreamSynchronize(*(gpu_resource_.get_stream_ptr())));
      }
    } 
    bp_loop++;

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
  int old_device = -1;
  CK_CUDA_THROW_(get_set_device(device_id_, &old_device));

  float* weight = (float*)malloc(weight_buff_.get_size());
  CK_CUDA_THROW_(cudaMemcpy(weight, weight_buff_.get_ptr_with_offset(0), weight_buff_.get_size(),
                            cudaMemcpyDeviceToHost));
  weight_stream.write(reinterpret_cast<char*>(weight), weight_buff_.get_size());
  free(weight);

  CK_CUDA_THROW_(get_set_device(old_device));

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
  int old_device = -1;
  CK_CUDA_THROW_(get_set_device(device_id_, &old_device));

  float* params = (float*)malloc(weight_buff_.get_size());
  params_stream.read(reinterpret_cast<char*>(params), weight_buff_.get_size());
  CK_CUDA_THROW_(cudaMemcpy(weight_buff_.get_ptr_with_offset(0), params, weight_buff_.get_size(),
                            cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(get_set_device(old_device));

  return;
}

void Network::download_params_to_host(float* weight) {
  int old_device = -1;
  CK_CUDA_THROW_(get_set_device(device_id_, &old_device));

  CK_CUDA_THROW_(cudaMemcpy(weight, weight_buff_.get_ptr_with_offset(0), weight_buff_.get_size(),
                            cudaMemcpyDeviceToHost));

  CK_CUDA_THROW_(get_set_device(old_device));

  return;
}

void Network::upload_params_to_device(float* params) {
  int old_device = -1;
  CK_CUDA_THROW_(get_set_device(device_id_, &old_device));
  CK_CUDA_THROW_(cudaMemcpy(weight_buff_.get_ptr_with_offset(0), params, weight_buff_.get_size(),
                            cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(get_set_device(old_device));

  return;
}

void Network::init_params(std::ofstream& out_stream) {
  for (auto layer : layers_) layer->init_params(out_stream);
}

float Network::get_loss() {
  float loss_host = 0.f;

  int old_device = -1;
  CK_CUDA_THROW_(get_set_device(device_id_, &old_device));

  CK_CUDA_THROW_(
      cudaMemcpy(&loss_host, loss_tensor_->get_ptr(), sizeof(float), cudaMemcpyDeviceToHost));

  CK_CUDA_THROW_(get_set_device(old_device));

  return loss_host;
}

void Network::exchange_wgrad() {
  if (gpu_resource_.get_nccl_ptr() != nullptr) {
    int old_device = -1;
    CK_CUDA_THROW_(get_set_device(device_id_, &old_device));

    CK_NCCL_THROW_(ncclAllReduce(
        (const void*)wgrad_buff_.get_ptr_with_offset(0), (void*)wgrad_buff_.get_ptr_with_offset(0),
        wgrad_buff_.get_num_elements(), ncclFloat, ncclSum, *(gpu_resource_.get_nccl_ptr()),
        *(gpu_resource_.get_stream_ptr())));

    CK_CUDA_THROW_(get_set_device(old_device));
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

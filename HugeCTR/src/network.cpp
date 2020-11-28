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

#include <layers/fully_connected_layer.hpp>
#include <layers/relu_layer.hpp>
#include <network.hpp>
#include <regularizers/no_regularizer.hpp>

namespace HugeCTR {

void conv_weight_gpu(size_t grid, size_t block, __half* dst, float* src, int elems,
                     cudaStream_t stream);

Network::Network(const std::shared_ptr<CPUResource>& cpu_resource,
                 const std::shared_ptr<GPUResource>& gpu_resource,
                 bool full_fp16,
                 bool enable_cuda_graph)
    : cpu_resource_(cpu_resource),
      gpu_resource_(gpu_resource),
      full_fp16_(full_fp16),
#ifdef NDEBUG
      enable_cuda_graph_(enable_cuda_graph),
#else
      enable_cuda_graph_(false),
#endif
      eval_graph_created_(false),
      train_fprop_graph_created_(false),
      train_bprop_graph_created_(false) {
}

void Network::update_params() {
  optimizer_->update();
  return;
}

void Network::conv_weight_() {
  CudaDeviceContext context(get_device_id());
  size_t elems = weight_tensor_half_.get_num_elements();
  if (weight_tensor_half_.get_num_elements() != weight_tensor_.get_num_elements())
    CK_THROW_(Error_t::WrongInput, "weight_buff_half_ != weight_buff");
  const size_t BLOCK = 256;
  size_t GRID = (elems - 1) / BLOCK + 1;
  GRID = GRID > 10 * gpu_resource_->get_sm_count() ? 10 * gpu_resource_->get_sm_count() : GRID;
  conv_weight_gpu(GRID, BLOCK, weight_tensor_half_.get_ptr(), weight_tensor_.get_ptr(), elems,
                  gpu_resource_->get_stream());
}

void Network::train(long long current_batchsize) {

  // forward
  if (full_fp16_) {
    conv_weight_();
  }

  if (enable_cuda_graph_) {
    if (!train_fprop_graph_created_) {
      CK_CUDA_THROW_(
          cudaStreamBeginCapture(gpu_resource_->get_stream(), cudaStreamCaptureModeRelaxed));
      for (auto& layer : layers_) {
        layer->fprop(true);
      }
      CK_CUDA_THROW_(cudaStreamEndCapture(gpu_resource_->get_stream(), &train_fprop_graph_));
      CK_CUDA_THROW_(
          cudaGraphInstantiate(&train_fprop_instance_, train_fprop_graph_, NULL, NULL, 0));
      train_fprop_graph_created_ = true;
    }
    CK_CUDA_THROW_(cudaGraphLaunch(train_fprop_instance_, gpu_resource_->get_stream()));
  } else {
    for (auto& layer : layers_) {
      layer->fprop(true);
    }
  }

  loss_->compute(true, current_batchsize);

  if (enable_cuda_graph_) {
    if (!train_bprop_graph_created_) {
      CK_CUDA_THROW_(
          cudaStreamBeginCapture(gpu_resource_->get_stream(), cudaStreamCaptureModeRelaxed));

      // backward
      for (auto it = layers_.rbegin(); it != layers_.rend(); it++) {
        (*it)->bprop();
      }
      CK_CUDA_THROW_(cudaStreamEndCapture(gpu_resource_->get_stream(), &train_bprop_graph_));
      CK_CUDA_THROW_(
          cudaGraphInstantiate(&train_bprop_instance_, train_bprop_graph_, NULL, NULL, 0));
      train_bprop_graph_created_ = true;
    }
    CK_CUDA_THROW_(cudaGraphLaunch(train_bprop_instance_, gpu_resource_->get_stream()));
  } else {
    // backward
    for (auto it = layers_.rbegin(); it != layers_.rend(); it++) {
      (*it)->bprop();
    }
  }

  return;
}

void Network::eval() {
  if (enable_cuda_graph_) {
    if (!eval_graph_created_) {
      CK_CUDA_THROW_(
          cudaStreamBeginCapture(gpu_resource_->get_stream(), cudaStreamCaptureModeRelaxed));
      // forward
      for (auto& layer : layers_) {
        layer->fprop(false);
      }
      CK_CUDA_THROW_(cudaStreamEndCapture(gpu_resource_->get_stream(), &eval_graph_));
      CK_CUDA_THROW_(cudaGraphInstantiate(&eval_instance_, eval_graph_, NULL, NULL, 0));
      eval_graph_created_ = true;
    }
    CK_CUDA_THROW_(cudaGraphLaunch(eval_instance_, gpu_resource_->get_stream()));
  } else {
    // forward
    for (auto& layer : layers_) {
      layer->fprop(false);
    }
  }
  loss_->compute(false);

  return;
}

void Network::download_params_to_host(std::ofstream& weight_stream) {
  // forward
  CudaDeviceContext context(get_device_id());

  std::unique_ptr<char[]> weight(new char[weight_tensor_.get_size_in_bytes()]);
  CK_CUDA_THROW_(cudaMemcpy(weight.get(), weight_tensor_.get_ptr(),
                            weight_tensor_.get_size_in_bytes(), cudaMemcpyDeviceToHost));
  weight_stream.write(weight.get(), weight_tensor_.get_size_in_bytes());

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
    CK_THROW_(Error_t::WrongInput,
              std::string("Cannot open dense model file (reason: ") + std::strerror(errno) + ")");
  }
  CudaDeviceContext context(get_device_id());

  std::unique_ptr<char[]> params(new char[weight_tensor_.get_size_in_bytes()]);
  model_stream.read(params.get(), weight_tensor_.get_size_in_bytes());
  CK_CUDA_THROW_(cudaMemcpy(weight_tensor_.get_ptr(), params.get(),
                            weight_tensor_.get_size_in_bytes(), cudaMemcpyHostToDevice));
  model_stream.close();
  return;
}

void Network::download_params_to_host(float* weight) {
  CudaDeviceContext context(get_device_id());

  CK_CUDA_THROW_(cudaMemcpy(weight, weight_tensor_.get_ptr(), weight_tensor_.get_size_in_bytes(),
                            cudaMemcpyDeviceToHost));

  return;
}

void Network::upload_params_to_device(float* params) {
  CudaDeviceContext context(get_device_id());

  CK_CUDA_THROW_(cudaMemcpy(weight_tensor_.get_ptr(), params, weight_tensor_.get_size_in_bytes(),
                            cudaMemcpyHostToDevice));

  return;
}

void Network::init_params(const std::string& model_file_name) {
  std::ofstream out_stream(model_file_name, std::ofstream::binary);
  if (!out_stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Cannot open dense model file");
  }
  for (auto& layer : layers_) layer->init_params(out_stream, *cpu_resource_);
  out_stream.close();
}

void Network::copy_params(const Network& n) {
  assert(weight_tensor_.get_size_in_bytes() == n.weight_tensor_.get_size_in_bytes());
  CK_CUDA_THROW_(cudaMemcpy(weight_tensor_.get_ptr(), n.weight_tensor_.get_ptr(),
                            weight_tensor_.get_size_in_bytes(), cudaMemcpyDeviceToDevice));
}

float Network::get_loss() {
  float loss_host = 0.f;

  CudaDeviceContext context(get_device_id());

  CK_CUDA_THROW_(
      cudaMemcpy(&loss_host, loss_tensor_.get_ptr(), sizeof(float), cudaMemcpyDeviceToHost));

  return loss_host;
}

metrics::RawMetricMap Network::get_raw_metrics() const { return raw_metrics_; }

void Network::exchange_wgrad() {
  if (gpu_resource_->support_nccl()) {
    CudaDeviceContext context(get_device_id());
    if (full_fp16_) {
      CK_NCCL_THROW_(ncclAllReduce((const void*)wgrad_tensor_half_.get_ptr(),
                                   (void*)wgrad_tensor_half_.get_ptr(),
                                   wgrad_tensor_half_.get_num_elements(), ncclHalf, ncclSum,
                                   gpu_resource_->get_nccl(), gpu_resource_->get_stream()));
    } else {
      CK_NCCL_THROW_(ncclAllReduce((const void*)wgrad_tensor_.get_ptr(),
                                   (void*)wgrad_tensor_.get_ptr(), wgrad_tensor_.get_num_elements(),
                                   ncclFloat, ncclSum, gpu_resource_->get_nccl(),
                                   gpu_resource_->get_stream()));
    }
  } else {
    CK_THROW_(Error_t::IllegalCall, "cannot call exchange_wgrad with single GPU");
  }
}

}  // namespace HugeCTR

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

#include <layers/fully_connected_layer.hpp>
#include <layers/relu_layer.hpp>
#include <network.hpp>
#include <regularizers/no_regularizer.hpp>

namespace HugeCTR {

void conv_weight_gpu(size_t grid, size_t block, __half* dst, const float* src, int elems,
                     cudaStream_t stream);

Network::Network(const std::shared_ptr<CPUResource>& cpu_resource,
                 const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision,
                 bool enable_cuda_graph)
    : cpu_resource_(cpu_resource),
      gpu_resource_(gpu_resource),
      use_mixed_precision_(use_mixed_precision),
#ifdef NDEBUG
      enable_cuda_graph_(enable_cuda_graph)
#else
      enable_cuda_graph_(false)
#endif
{
}

void Network::update_params() {
  optimizer_->update();
  return;
}

void Network::conv_weight_(Tensor2<__half>& target, const Tensor2<float>& source) {
  CudaDeviceContext context(get_device_id());
  size_t elems = source.get_num_elements();
  if (target.get_num_elements() != source.get_num_elements())
    CK_THROW_(Error_t::WrongInput, "weight size of target != weight size of in");
  const size_t BLOCK = 256;
  size_t GRID = (elems - 1) / BLOCK + 1;
  GRID = GRID > 10 * gpu_resource_->get_sm_count() ? 10 * gpu_resource_->get_sm_count() : GRID;
  conv_weight_gpu(GRID, BLOCK, target.get_ptr(), source.get_ptr(), elems,
                  gpu_resource_->get_stream());
}

cudaEvent_t& Network::get_train_events(TrainState_t key) {
  if (train_events_.find(key) == train_events_.end()) {
    cudaEvent_t event;
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    train_events_[key] = event;
  }
  return train_events_[key];
}

template <typename LPtr>
void Network::prop_layers(const std::vector<LPtr>& layers, Network::GraphWrapper& graph,
                          bool use_graph, bool fprop, const cudaStream_t stream, bool train) {
  auto execute = [&layers, train](bool fprop) {
    if (fprop) {
      for (auto& layer : layers) {
        layer->fprop(train);
      }
    } else {
      for (auto it = layers.rbegin(); it != layers.rend(); it++) {
        (*it)->bprop();
      }
    }
  };

  if (!use_graph) {
    execute(fprop);
    return;
  }

  bool do_capture;
#ifdef ENABLE_PROFILING
  if (profiler_init_cuda_graph_this_iter()) {
    do_capture = !graph.initialized_with_profiling;
    graph.initialized = false;
    graph.initialized_with_profiling = true;
  } else {
    do_capture = !graph.initialized;
    graph.initialized = true;
    graph.initialized_with_profiling = false;
  }
#else
  do_capture = !graph.initialized;
  graph.initialized = true;
#endif

  if (do_capture) {
    CK_CUDA_THROW_(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    execute(fprop);
    CK_CUDA_THROW_(cudaStreamEndCapture(stream, &graph.graph));
    CK_CUDA_THROW_(cudaGraphInstantiate(&graph.graph_exec, graph.graph, NULL, NULL, 0));
  }
  CK_CUDA_THROW_(cudaGraphLaunch(graph.graph_exec, stream));
}

void Network::train(long long current_batchsize) {
  // forward
  if (use_mixed_precision_) {
    conv_weight_(train_weight_tensor_half_, train_weight_tensor_);
  }
  prop_layers(train_layers_, train_fprop_graph_, enable_cuda_graph_, true,
              gpu_resource_->get_stream());
  train_loss_->compute(true, current_batchsize);
  prop_layers(train_layers_, train_bprop_graph_, enable_cuda_graph_, false,
              gpu_resource_->get_stream());
  return;
}

TrainState Network::train(long long current_batchsize, std::function<void()> exchange_wgrad,
                          TrainState state) {
  auto stream = gpu_resource_->get_stream();

  switch (state.state) {
    case TrainState_t::Init:
      if (use_mixed_precision_) {
        conv_weight_(train_weight_tensor_half_, train_weight_tensor_);
      }
      break;
    case TrainState_t::BottomMLPFprop:
      prop_layers(bottom_layers_, bottom_train_fprop_graph_, enable_cuda_graph_, true, stream);
      break;
    case TrainState_t::TopMLPFprop:
      prop_layers(top_layers_, train_fprop_graph_, enable_cuda_graph_, true, stream);
      train_loss_->compute(true, current_batchsize);
      break;
    case TrainState_t::TopMLPBprop:
      prop_layers(top_layers_, train_bprop_graph_, enable_cuda_graph_, false, stream);
      break;
    case TrainState_t::BottomMLPBprop:
      prop_layers(bottom_layers_, bottom_train_bprop_graph_, enable_cuda_graph_, false, stream);
      break;
    case TrainState_t::MLPExchangeWgrad:
      exchange_wgrad();
      break;
    case TrainState_t::MLPUpdate:
      update_params();
      break;
    case TrainState_t::Finalize:
      break;
    default:
      CK_THROW_(Error_t::InvalidEnv, "network train reach invalid status");
  }

  cudaEvent_t& event = get_train_events(state.state);
  CK_CUDA_THROW_(cudaEventRecord(event, stream));
  state.event = &event;
  return state;
}

void Network::eval(long long current_batchsize) {
  prop_layers(evaluate_layers_, eval_graph_, enable_cuda_graph_, true, gpu_resource_->get_stream(),
              false);
  evaluate_loss_->compute(false, current_batchsize);
}
void Network::predict() {
  prop_layers(evaluate_layers_, predict_graph_, enable_cuda_graph_, true,
              gpu_resource_->get_stream(), false);
}

void Network::download_params_to_host(std::ofstream& weight_stream) {
  // forward
  CudaDeviceContext context(get_device_id());

  std::unique_ptr<char[]> weight(new char[train_weight_tensor_.get_size_in_bytes()]);
  CK_CUDA_THROW_(cudaMemcpy(weight.get(), train_weight_tensor_.get_ptr(),
                            train_weight_tensor_.get_size_in_bytes(), cudaMemcpyDeviceToHost));
  weight_stream.write(weight.get(), train_weight_tensor_.get_size_in_bytes());

  return;
}

void Network::download_opt_states_to_host(std::ofstream& opt_states_stream) {
  // forward
  CudaDeviceContext context(get_device_id());

  size_t dst_size_in_byte =
      use_mixed_precision_ ? opt_tensor_half_.get_size_in_bytes() : opt_tensor_.get_size_in_bytes();
  std::unique_ptr<char[]> h_opt_states(new char[dst_size_in_byte]);

  void* src =
      use_mixed_precision_ ? (void*)opt_tensor_half_.get_ptr() : (void*)opt_tensor_.get_ptr();
  CK_CUDA_THROW_(cudaMemcpy(h_opt_states.get(), src, dst_size_in_byte, cudaMemcpyDeviceToHost));

  opt_states_stream.write(h_opt_states.get(), dst_size_in_byte);
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
    CK_THROW_(Error_t::WrongInput,
              std::string("Cannot open dense model file (reason: ") + std::strerror(errno) + ")");
  }
  CudaDeviceContext context(get_device_id());

  std::unique_ptr<char[]> params(new char[train_weight_tensor_.get_size_in_bytes()]);
  model_stream.read(params.get(), train_weight_tensor_.get_size_in_bytes());
  CK_CUDA_THROW_(cudaMemcpy(train_weight_tensor_.get_ptr(), params.get(),
                            train_weight_tensor_.get_size_in_bytes(), cudaMemcpyHostToDevice));
  model_stream.close();
  return;
}

void Network::upload_params_to_device_inference(const std::string& model_file) {
  std::ifstream model_stream(model_file, std::ifstream::binary);
  if (!model_stream.is_open()) {
    CK_THROW_(Error_t::WrongInput,
              std::string("Cannot open dense model file (reason: ") + std::strerror(errno) + ")");
  }
  CudaDeviceContext context(get_device_id());

  std::unique_ptr<char[]> params(new char[evaluate_weight_tensor_.get_size_in_bytes()]);
  model_stream.read(params.get(), evaluate_weight_tensor_.get_size_in_bytes());
  CK_CUDA_THROW_(cudaMemcpyAsync(evaluate_weight_tensor_.get_ptr(), params.get(),
                                 evaluate_weight_tensor_.get_size_in_bytes(),
                                 cudaMemcpyHostToDevice, gpu_resource_->get_stream()));
  model_stream.close();
  if (use_mixed_precision_) {
    conv_weight_(evaluate_weight_tensor_half_, evaluate_weight_tensor_);
  }
  return;
}

void Network::download_params_to_host(float* weight) {
  CudaDeviceContext context(get_device_id());

  CK_CUDA_THROW_(cudaMemcpy(weight, train_weight_tensor_.get_ptr(),
                            train_weight_tensor_.get_size_in_bytes(), cudaMemcpyDeviceToHost));

  return;
}

void Network::upload_params_to_device(float* params) {
  CudaDeviceContext context(get_device_id());

  CK_CUDA_THROW_(cudaMemcpy(train_weight_tensor_.get_ptr(), params,
                            train_weight_tensor_.get_size_in_bytes(), cudaMemcpyHostToDevice));

  return;
}

void Network::upload_opt_states_to_device(char* h_opt_states) {
  CudaDeviceContext context(get_device_id());

  size_t src_size_in_byte =
      use_mixed_precision_ ? opt_tensor_half_.get_size_in_bytes() : opt_tensor_.get_size_in_bytes();

  void* dst =
      use_mixed_precision_ ? (void*)opt_tensor_half_.get_ptr() : (void*)opt_tensor_.get_ptr();

  CK_CUDA_THROW_(cudaMemcpy(dst, h_opt_states, src_size_in_byte, cudaMemcpyHostToDevice));

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

  CudaDeviceContext context(get_device_id());
  CK_CUDA_THROW_(cudaMemcpyAsync(&loss_host, train_loss_tensor_.get_ptr(), sizeof(float),
                                 cudaMemcpyDeviceToHost, gpu_resource_->get_stream()));
  CK_CUDA_THROW_(cudaStreamSynchronize(gpu_resource_->get_stream()));
  return loss_host;
}

metrics::RawMetricMap Network::get_raw_metrics() const { return raw_metrics_; }

void Network::exchange_wgrad() {
  CudaDeviceContext context(get_device_id());
  if (use_mixed_precision_) {
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
}

void Network::copy_weights_from_train_layers_to_evaluate_layers() {
  CudaDeviceContext context(get_device_id());
  cudaMemcpyAsync(evaluate_weight_tensor_.get_ptr(), train_weight_tensor_.get_ptr(),
                  train_weight_tensor_.get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                  gpu_resource_->get_stream());

  if (use_mixed_precision_) {
    conv_weight_(evaluate_weight_tensor_half_, evaluate_weight_tensor_);
  }
}

}  // namespace HugeCTR

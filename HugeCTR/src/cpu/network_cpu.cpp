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

#include <cpu/network_cpu.hpp>

namespace HugeCTR {

NetworkCPU::NetworkCPU(const std::shared_ptr<CPUResource>& cpu_resource, bool use_mixed_precision)
    : cpu_resource_(cpu_resource), use_mixed_precision_(use_mixed_precision) {}

void NetworkCPU::conv_weight_(Tensor2<__half>& target, const Tensor2<float>& source) {
  size_t elems = source.get_num_elements();
  if (target.get_num_elements() != source.get_num_elements())
    HCTR_OWN_THROW(Error_t::WrongInput, "weight size of target != weight size of in");
  __half* h_target = target.get_ptr();
  const float* h_source = source.get_ptr();
  for (size_t i = 0; i < elems; i++) {
    h_target[i] = __float2half(h_source[i]);
  }
}

void NetworkCPU::predict() {
  if (use_mixed_precision_) {
    conv_weight_(weight_tensor_half_, weight_tensor_);
  }
  // forward
  for (auto& layer : layers_) {
    layer->fprop(false);
  }
  return;
}

void NetworkCPU::load_params_from_model(const std::string& model_file) {
  std::ifstream model_stream(model_file, std::ifstream::binary);
  if (!model_stream.is_open()) {
    std::ostringstream os;
    os << "Cannot open dense model file (reason: " << std::strerror(errno) << ')';
    HCTR_OWN_THROW(Error_t::WrongInput, os.str());
  }
  model_stream.read((char*)weight_tensor_.get_ptr(), weight_tensor_.get_size_in_bytes());
  model_stream.close();
  return;
}

void NetworkCPU::initialize() {
  for (auto& layer : layers_) {
    layer->initialize();
  }
}

}  // namespace HugeCTR

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

#include <layer.hpp>
#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

void Layer::init_params(const curandGenerator_t& generator) {
  std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> buff =
      GeneralBuffer2<CudaHostAllocator>::create();
  std::shared_ptr<BufferBlock2<float>> block = buff->create_block<float>();

  Tensors2<float> weight_cpu_tensors;
  for (const Tensor2<float>& weight : weights_) {
    Tensor2<float> tensor;
    block->reserve(weight.get_dimensions(), &tensor);
    weight_cpu_tensors.push_back(tensor);
  }

  buff->allocate();

  std::vector<std::unique_ptr<DataSimulator>> simulators;
  for (int index = 0; index < static_cast<int>(initializer_types_.size()); ++index) {
    switch (initializer_types_[index]) {
      case Initializer_t::Uniform: {
        simulators.push_back(get_uniform_initializer(index));
        break;
      }
      case Initializer_t::XavierNorm: {
        simulators.push_back(get_xavier_norm_initializer(index));
        break;
      }
      case Initializer_t::XavierUniform: {
        simulators.push_back(get_xavier_uniform_initializer(index));
        break;
      }
      case Initializer_t::Zero: {
        simulators.push_back(get_zero_initializer(index));
        break;
      }
      case Initializer_t::Default: {
        simulators.push_back(get_default_initializer(index));
        break;
      }
      default: {
        CK_THROW_(Error_t::OutOfBound, "Not supported initializer.");
        break;
      }
    }
  }

  for (size_t i = 0; i < weights_.size(); ++i) {
    simulators[i % simulators.size()]->fill(weight_cpu_tensors[i], generator);
    CK_CUDA_THROW_(cudaMemcpyAsync(weights_[i].get_ptr(), weight_cpu_tensors[i].get_ptr(),
                                   weights_[i].get_size_in_bytes(), cudaMemcpyHostToDevice,
                                   get_gpu().get_stream()));
  }
}

}  // namespace HugeCTR

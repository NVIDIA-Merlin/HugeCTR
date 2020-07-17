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

#include "HugeCTR/include/layer.hpp"

#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

void Layer::init_params(std::ofstream& out_stream) {
  std::vector<float> initializer = std::move(get_initializer());
  if (initializer.empty()) return;

  size_t size_in_byte = initializer.size() * sizeof(float);
  out_stream.write(reinterpret_cast<char*>(&initializer.front()), size_in_byte);
}

std::vector<float> Layer::get_initializer() {
  size_t elements = 0;
  for (const auto& weight : weights_){
    elements += weight->get_num_elements();
  }
  std::vector<float> initializer(elements, 0.f);

  std::vector<std::unique_ptr<DataSimulator<float>>> simulators;
  for (int index = 0; index < initializer_types_.size(); ++index) {
    switch (initializer_types_[index]) {
      case Initializer_t::Uniform : {
        simulators.push_back(get_uniform_initializer(index));
        break;
      }
      case Initializer_t::XavierNorm : {
        simulators.push_back(get_xavier_norm_initializer(index));
        break;
      }
      case Initializer_t::XavierUniform : {
        simulators.push_back(get_xavier_uniform_initializer(index));
        break;
      }
      case Initializer_t::Zero : {
        simulators.push_back(get_zero_initializer(index));
        break;
      }
      case Initializer_t::Default : {
        simulators.push_back(get_default_initializer(index));
        break;
      }
      default : {
        CK_THROW_(Error_t::OutOfBound, "Not supported initializer.");
        break;
      }
    }
  }

  size_t current_offset = 0;
  for (size_t w = 0; w < weights_.size(); ++w){
    for (size_t j = 0; j < (weights_[w])->get_num_elements(); ++j){
      initializer[j + current_offset] = simulators[w % simulators.size()]->get_num();
    }
    current_offset += (weights_[w])->get_num_elements();
  }

  for (auto& simu : simulators)
    simu.reset(nullptr);

  return initializer;
}

}  // namespace HugeCTR

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

#include "HugeCTR/include/inference/session_inference.hpp"

#include <iostream>
#include <vector>
namespace HugeCTR {

InferenceSession::InferenceSession(const std::string& config_file, cudaStream_t stream)
    : config_(read_json_file(config_file)),
      parser_(config_),
      inference_parser_(config_),
      resource_manager(ResourceManager::create({{0}}, 0)) {
  try {
    Network* network;
    parser_.create_pipeline(inference_parser_, row_, embeddingvector_, &embedding_, &network,
                            resource_manager);
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  return;
}  // namespace HugeCTR

InferenceSession::~InferenceSession() {}

void InferenceSession::predict(float* dense, int* row, float* embeddingvector, float* output,
                               int numofsample) {}
}  // namespace HugeCTR

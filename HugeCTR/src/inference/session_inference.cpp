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
session_inference::session_inference(std::string model_name) { model_name = model_name; }

session_inference::~session_inference() {}

void session_inference::predict(float* dense, int* row, float* embeddingvector, float* output,
                                int numofsample) {}
}  // namespace HugeCTR

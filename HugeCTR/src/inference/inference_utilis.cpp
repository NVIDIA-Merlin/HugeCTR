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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optimizer.hpp>
#include <parser.hpp>
#include <session.hpp>
#include <utils.hpp>
#include <inference/parameter_server.hpp>
#include <inference/inference_utils.hpp>

namespace HugeCTR {
template <typename TypeHashKey>
HugectrUtility<TypeHashKey>::HugectrUtility() {}

template <typename TypeHashKey>
HugectrUtility<TypeHashKey>::~HugectrUtility() {}

template <typename TypeHashKey>
HugectrUtility<TypeHashKey>* HugectrUtility<TypeHashKey>::Create_Embedding(INFER_TYPE Infer_type, const nlohmann::json& model_config) {
  HugectrUtility<TypeHashKey>* embedding;

  switch (Infer_type) {
    case TRITON:
      embedding = new parameter_server<TypeHashKey>("TRITON", model_config);
      break;
    default:
      std::cout << "wrong type!" << std::endl;
      return NULL;
  }

  return embedding;
}

}  // namespace HugeCTR

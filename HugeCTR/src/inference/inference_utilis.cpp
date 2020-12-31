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

#include "HugeCTR/include/inference/embedding_cache.hpp"
#include "HugeCTR/include/inference/inference_utils.hpp"

namespace HugeCTR {
template <typename T>
HugectrUtility<T>::HugectrUtility() {}
template <typename T>
HugectrUtility<T>::~HugectrUtility() {}
template <typename T>
HugectrUtility<T>* HugectrUtility<T>::Create_Embedding(INFER_TYPE Infer_type, std::string& config) {
  HugectrUtility<T>* embedding;

  switch (Infer_type) {
    case TRITON:
      embedding = new embedding_cache<T>("TRITON");
      break;
    default:
      std::cout << "wrong type!" << std::endl;
      return NULL;
  }

  return embedding;
}

}  // namespace HugeCTR

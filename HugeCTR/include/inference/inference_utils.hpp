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

#pragma once
#include <common.hpp>
#include <embedding.hpp>
#include <metrics.hpp>
#include <network.hpp>
#include <parser.hpp>
#include <string>
#include <thread>
#include <utility>

namespace HugeCTR {

template <typename TypeHashKey>
class HugectrUtility {
 public:
  HugectrUtility();
  virtual ~HugectrUtility();
  virtual void look_up(const TypeHashKey* embeddingcolumns, size_t length, float* embeddingoutputvector, cudaStream_t stream) = 0;
  static HugectrUtility<TypeHashKey>* Create_Embedding(INFER_TYPE Infer_type, const nlohmann::json& model_config);
};

}  // namespace HugeCTR
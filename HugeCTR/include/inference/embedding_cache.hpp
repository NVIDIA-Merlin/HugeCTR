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
#include <vector>

#include "HugeCTR/include/inference/hugectrmodel.hpp"
#include "HugeCTR/include/inference/inference_utils.hpp"

namespace HugeCTR {

template <typename T>
class embedding_cache : public HugectrUtility<T> {
 public:
  embedding_cache(std::string model_name);
  virtual ~embedding_cache();

  void look_up(T* embeddingcolumns, T* length, float* embeddingoutputvector);

 private:
  std::string model_name;
};

}  // namespace HugeCTR
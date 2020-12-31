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
#pragma once
#include <common.hpp>
#include <embedding.hpp>
#include <metrics.hpp>
#include <network.hpp>
#include <parser.hpp>
#include <string>
#include <thread>
#include <utility>

#include "HugeCTR/include/embeddings/embedding.hpp"

namespace HugeCTR {
enum INFER_TYPE { TRITON, OTHER };

class HugeCTRModel {
 public:
  HugeCTRModel();
  virtual ~HugeCTRModel();
  virtual void predict(float* d_dense, void* h_embeddingcolumns, int* d_row_ptrs,
                      float* d_embeddingvectors, float* d_output, int num_samples) = 0;
  static HugeCTRModel* load_model(INFER_TYPE Infer_type, std::string& config);
};

}  // namespace HugeCTR

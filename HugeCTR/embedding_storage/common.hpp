/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <map>
#include <string>
#include <vector>

#include "HugeCTR/core/buffer.hpp"
#include "HugeCTR/embedding/common.hpp"
#include "HugeCTR/include/optimizer.hpp"

namespace embedding {
using core::CoreResourceManager;
using core::DataType;
using core::Device;
using core::DeviceType;
using core::GetBuffer;
using core::GetBufferBlock;
using core::Shape;
using core::Tensor;
using core::TensorList;
using core::TensorScalarType;

struct UniformParams {
  float up_bound;
};
struct SinusoidalParams {
  int ev_size;
  int max_sequence_len;
};

struct EmbeddingTableInitParams {
  HugeCTR::Initializer_t initializer_type;
  UniformParams uniform_params;
  SinusoidalParams sinus_params;
};

struct EmbeddingTableParam {
  int table_id;
  int max_vocabulary_size;  // -1 means dynamic
  int ev_size;
  int64_t min_key;
  int64_t max_key;

  HugeCTR::OptParams opt_param;
  EmbeddingTableInitParams init_param;
};
}  // namespace embedding

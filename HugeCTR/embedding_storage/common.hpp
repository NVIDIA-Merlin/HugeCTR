/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <core/buffer.hpp>
#include <embedding/common.hpp>
#include <map>
#include <optimizer.hpp>
#include <string>
#include <vector>

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

struct InitParams {
  HugeCTR::Initializer_t initializer_type;
  UniformParams uniform_params;
  SinusoidalParams sinusoidal_params;

  InitParams() : initializer_type{HugeCTR::Initializer_t::Default} {
    uniform_params.up_bound = 0.f;
    sinusoidal_params.ev_size = 0;
    sinusoidal_params.max_sequence_len = 0;
  }

  // Better to use overload. But Initailizer_t is coupled and used in other
  // places.
  explicit InitParams(int ev_size,
                      HugeCTR::Initializer_t initializer_type = HugeCTR::Initializer_t::Default,
                      float up_bound = -1, int max_sequence_len = -1) {
    this->initializer_type = initializer_type;
    if (initializer_type == HugeCTR::Initializer_t::Uniform) {
      HCTR_CHECK_HINT(up_bound > 0, "initialize type uniform should specify up_bound");
      this->uniform_params.up_bound = up_bound;
    }
    if (initializer_type == HugeCTR::Initializer_t::Sinusoidal) {
      HCTR_CHECK_HINT(max_sequence_len > 0,
                      "initialize type uniform should specify max_sequence_len");
      this->sinusoidal_params.ev_size = ev_size;
      this->sinusoidal_params.max_sequence_len = max_sequence_len;
    }
  }
};

struct EmbeddingTableParam {
  int table_id;
  int max_vocabulary_size;  // -1 means dynamic
  int ev_size;

  HugeCTR::OptParams opt_param;
  InitParams init_param;

  EmbeddingTableParam() = default;

  EmbeddingTableParam(int table_id, int max_vocabulary_size, int ev_size,
                      HugeCTR::OptParams opt_param, InitParams init_param) {
    this->table_id = table_id;
    this->max_vocabulary_size = max_vocabulary_size;
    this->ev_size = ev_size;
    this->opt_param = opt_param;
    this->init_param = init_param;
  }
};
}  // namespace embedding

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

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <hps/inference_utils.hpp>
#include <limits>

namespace HugeCTR {

template <typename InT, typename OutT>
class Dequantize {
 public:
  Dequantize();
  void dequantize(InT* input, OutT* output, OutT* scale, size_t batch_size,
                  size_t emb_vec_size) const;
};
}  // namespace HugeCTR
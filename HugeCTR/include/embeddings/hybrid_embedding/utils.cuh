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

#include <cuda_runtime.h>

namespace HugeCTR {
namespace hybrid_embedding {

__global__ void offsets_kernel(const uint32_t* indices, uint32_t* indices_offsets,
                               uint32_t num_instances, uint32_t multiplier);

__global__ void model_id_kernel(const uint32_t* indices_offsets, uint32_t* src_model_id,
                                const uint32_t* d_num_elements);

template <typename dtype, typename stype>
__global__ void modulo_kernel(dtype* buffer, const stype* d_num_elements, dtype divisor);

}  // namespace hybrid_embedding
}  // namespace HugeCTR

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

#include <utils.cuh>

namespace HugeCTR {

void conv_weight_gpu(size_t grid, size_t block, __half* dst, float* src, int elems,
                     cudaStream_t stream) {
  convert_array<<<grid, block, 0, stream>>>(dst, src, elems);
}


}  // end namespace HugeCTR

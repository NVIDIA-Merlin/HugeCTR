/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <utils.hpp>

namespace HugeCTR {

template <typename TIN, typename TOUT>
void convert_array_on_device(TOUT *out, const TIN *in, size_t num_elements,
                             const cudaStream_t &stream) {
  if (num_elements > 0) {
    convert_array<<<(num_elements - 1) / 1024 + 1, 1024, 0, stream>>>(out, in, num_elements);
  }
}

template void convert_array_on_device<long long, int>(int *, const long long *, size_t,
                                                      const cudaStream_t &);
template void convert_array_on_device<unsigned int, int>(int *, const unsigned int *, size_t,
                                                         const cudaStream_t &);
template void convert_array_on_device<float, float>(float *, const float *, size_t,
                                                    const cudaStream_t &);
template void convert_array_on_device<float, __half>(__half *, const float *, size_t,
                                                     const cudaStream_t &);
template void convert_array_on_device<__half, float>(float *, const __half *, size_t,
                                                     const cudaStream_t &);

template <typename TypeKey>
void data_to_unique_categories(TypeKey *value, const TypeKey *rowoffset,
                               const TypeKey *emmbedding_offsets, int num_tables,
                               int num_rowoffsets, const cudaStream_t &stream) {
  constexpr size_t block_size = 256;
  size_t grid_size = (num_rowoffsets - 1) / block_size + 1;
  unique_key_kernels::data_to_unique_categories_kernel<<<grid_size, block_size, 0, stream>>>(
      value, rowoffset, emmbedding_offsets, num_tables, num_rowoffsets);
}

template void data_to_unique_categories<long long>(long long *, const long long *,
                                                   const long long *, int, int,
                                                   const cudaStream_t &);

template void data_to_unique_categories<unsigned int>(unsigned int *, const unsigned int *,
                                                      const unsigned int *, int, int,
                                                      const cudaStream_t &);

template <typename TypeKey>
void data_to_unique_categories(TypeKey *value, const TypeKey *emmbedding_offsets, int num_tables,
                               int nnz, const cudaStream_t &stream) {
  constexpr size_t block_size = 256;
  size_t grid_size = std::min(4096ul, (nnz - 1) / block_size + 1);
  if (num_tables % 2 == 0 && sizeof(TypeKey) == 4) {
    unique_key_kernels::
        data_to_unique_categories_align2_kernel<<<grid_size, block_size, 0, stream>>>(
            value, emmbedding_offsets, num_tables / 2, nnz / 2);
  } else {
    unique_key_kernels::data_to_unique_categories_kernel<<<grid_size, block_size, 0, stream>>>(
        value, emmbedding_offsets, num_tables, nnz);
  }
}

template void data_to_unique_categories<long long>(long long *, const long long *, int, int,
                                                   const cudaStream_t &);

template void data_to_unique_categories<unsigned int>(unsigned int *, const unsigned int *, int,
                                                      int, const cudaStream_t &);

template <typename T>
__global__ void inc_var_cuda(T *x) {
  if (blockIdx.x == 0 and threadIdx.x == 0){
    (*x)++;
  }
}

template <typename T>
void inc_var(T *x, cudaStream_t stream){
  inc_var_cuda<<<1, 32, 0, stream>>>(x);
}

template void inc_var<size_t>(size_t *x, cudaStream_t stream);

}  // namespace HugeCTR

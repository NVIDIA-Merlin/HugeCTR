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

#include <utest/embedding_collection/reference_embedding.hpp>
#include <utils.cuh>

__global__ void copy_float_emb_vec_ptrs_data_kernel(const float **emb_vec_ptrs, float *data,
                                                    size_t num_keys, int vec_size

) {
  CUDA_1D_KERNEL_LOOP(i, num_keys * vec_size) {
    int sample_id = i / vec_size;
    int vec_id = i % vec_size;

    data[i] = emb_vec_ptrs[sample_id][vec_id];
  }
}

__global__ void combiner_kernel(float *data, const int *division_ptr,
                                const int *ev_start_indices_ptr, const int *ev_size_ptr,
                                int num_bucket, int max_ev_size) {
  CUDA_1D_KERNEL_LOOP(i, num_bucket * max_ev_size) {
    int sample_id = i / max_ev_size;
    int vec_id = i % max_ev_size;

    int ev_size = ev_size_ptr[sample_id];
    if (vec_id < ev_size) {
      int division = division_ptr[sample_id];
      int ev_start_indices = ev_start_indices_ptr[sample_id];
      data[ev_start_indices + vec_id] /= division;
    }
  }
}

template <typename EmbType>
__global__ void copy_to_float_bufer(const EmbType *data, float *dst_data, int ev_start_indices,
                                    int num_elements) {
  CUDA_1D_KERNEL_LOOP(i, num_elements) {
    dst_data[ev_start_indices + i] = HugeCTR::TypeConvertFunc<float, EmbType>::convert(data[i]);
  }
}

void copy_float_emb_vec_ptrs_data(const core23::Tensor &float_emb_vec_ptrs, core23::Tensor &data,
                                  size_t num_keys, int vec_size, cudaStream_t stream) {
  copy_float_emb_vec_ptrs_data_kernel<<<num_keys * vec_size / 256 + 1, 256, 0, stream>>>(
      (const float **)float_emb_vec_ptrs.data(), data.data<float>(), num_keys, vec_size);
}

void combiner_func(const CombinerData &combiner_data, core23::Tensor &data, cudaStream_t stream) {
  combiner_kernel<<<combiner_data.num_bucket * combiner_data.max_ev_size / 256 + 1, 256, 0,
                    stream>>>(data.data<float>(), combiner_data.combiner_division.data<int>(),
                              combiner_data.ev_start_indices.data<int>(),
                              combiner_data.ev_size.data<int>(), combiner_data.num_bucket,
                              combiner_data.max_ev_size);
}

void copy_to_float_bufer(const core23::Tensor &data, core23::Tensor &float_buffer,
                         int ev_start_indices, cudaStream_t stream) {
  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(data.data_type().type(), EmbType, [&] {
    copy_to_float_bufer<<<data.num_elements() / 256 + 1, 256, 0, stream>>>(
        data.data<EmbType>(), float_buffer.data<float>(), ev_start_indices, data.num_elements());
  });
}

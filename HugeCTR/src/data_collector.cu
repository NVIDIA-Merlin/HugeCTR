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

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_collector.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/tensor.hpp"

namespace HugeCTR {

template <typename TypeComp>
__global__ void split_kernel__(int batchsize, float* label_ptr, int label_dim, TypeComp* dense_ptr,
                               int dense_dim, const float* label_dense) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < batchsize * (label_dim + dense_dim)) {
    const int in_col = idx % (label_dim + dense_dim);
    const int in_row = idx / (label_dim + dense_dim);
    const int out_row = in_row;
    if (in_col < label_dim) {
      const int out_col = in_col;
      label_ptr[out_row * label_dim + out_col] = label_dense[idx];
    } else {
      const int out_col = in_col - label_dim;
      dense_ptr[out_row * dense_dim + out_col] = label_dense[idx];
    }
  }
  return;
}

template <typename TypeComp>
void split(std::shared_ptr<Tensor<float>> label_tensor,
           std::shared_ptr<Tensor<TypeComp>> dense_tensor,
           const std::shared_ptr<GeneralBuffer<float>> label_dense_buffer, cudaStream_t stream) {
  // check the input size
  assert(label_tensor->get_dims()[0] == dense_tensor->get_dims()[0]);
  assert(label_tensor->get_num_elements() + dense_tensor->get_num_elements() ==
         label_dense_buffer->get_num_elements());

  const int batchsize = label_tensor->get_dims()[0];
  const int label_dim = label_tensor->get_dims()[1];
  const int dense_dim = dense_tensor->get_dims()[1];

  const int BLOCK_DIM = 256;
  const int GRID_DIM = (label_dense_buffer->get_num_elements() - 1) / BLOCK_DIM + 1;
  assert(dense_dim >= 0 || "dense_dim should be >= 0");

  if (dense_dim > 0) {
    split_kernel__<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
        batchsize, label_tensor->get_ptr(), label_dim, dense_tensor->get_ptr(), dense_dim,
        label_dense_buffer->get_ptr_with_offset(0));
  } else if (dense_dim == 0) {
    split_kernel__<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(batchsize, label_tensor->get_ptr(),
                                                       label_dim, (TypeComp*)0, 0,
                                                       label_dense_buffer->get_ptr_with_offset(0));

  } else {
    CK_THROW_(Error_t::WrongInput, "dense_dim < 0");
  }

  return;
}

template void split<float>(std::shared_ptr<Tensor<float>> label_tensor,
                           std::shared_ptr<Tensor<float>> dense_tensor,
                           const std::shared_ptr<GeneralBuffer<float>> label_dense_buffer,
                           cudaStream_t stream);

template void split<__half>(std::shared_ptr<Tensor<float>> label_tensor,
                            std::shared_ptr<Tensor<__half>> dense_tensor,
                            const std::shared_ptr<GeneralBuffer<float>> label_dense_buffer,
                            cudaStream_t stream);

}  // namespace HugeCTR

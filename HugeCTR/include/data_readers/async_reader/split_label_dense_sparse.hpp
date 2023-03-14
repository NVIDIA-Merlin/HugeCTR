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

#include <core23/tensor.hpp>
#include <tensor2.hpp>
namespace HugeCTR {
template <typename DenseType, typename SparseType>
void split_3_way(Tensor2<float> label_tensor_per_dev, Tensor2<DenseType> dense_tensor_per_dev,
                 Tensor2<SparseType> sparse_tensor, Tensor2<int> label_dense_sparse_buffer,
                 size_t local_idx_start, size_t local_idx_end, cudaStream_t stream);

template <typename DenseType, typename SparseType>
void split_3_way(core23::Tensor& label_tensor_per_dev, core23::Tensor& dense_tensor_per_dev,
                 core23::Tensor& sparse_tensor, const core23::Tensor& label_dense_sparse_buffer,
                 size_t local_idx_start, size_t local_idx_end, cudaStream_t stream);

}  // namespace HugeCTR

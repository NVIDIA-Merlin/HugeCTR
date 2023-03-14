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

#include <cassert>
#include <data_readers/multi_hot/split_batch.hpp>

namespace HugeCTR {
template <bool ISFLOAT = true>
struct DenseOp_t {
  __host__ __device__ __forceinline__ float operator()(const int* in) { return 0.f; }
  DenseOp_t() = default;
};
template <>
struct DenseOp_t<true> {
  __host__ __device__ __forceinline__ float operator()(const int* in) {
    return *reinterpret_cast<const float*>(in);
  }
};
template <>
struct DenseOp_t<false> {
  __host__ __device__ __forceinline__ float operator()(const int* in) {
    return static_cast<float>(logf(*in + 1.f));
  }
};

using int_dense_op_t = DenseOp_t<false>;
using float_dense_op_t = DenseOp_t<true>;

template <typename DenseType, typename SparseType, typename DenseOp>
__global__ void split_feat_major_kernel(float* __restrict label, int label_dim,
                                        DenseType* __restrict dense, int dense_dim,
                                        SparseType** __restrict sparse_tensors, int sparse_dim,
                                        const int* __restrict label_dense_sparse,
                                        const int* __restrict bucket_ids,
                                        const int* __restrict bucket_positions,
                                        const int* __restrict max_hotnesses, uint32_t batch_size,
                                        uint32_t sample_dim, DenseOp dop) {
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch_size * sample_dim;
       idx += blockDim.x * gridDim.x) {
    const uint32_t row = idx / sample_dim;
    const uint32_t col = idx - row * sample_dim;

    if (col < label_dim)  // store in label tensor
    {
      auto col_data = label_dense_sparse[idx];  // Load column
      label[row * label_dim + col] = static_cast<float>(col_data);
    } else if (col < label_dim + dense_dim)  // store in dense tensor
    {
      const auto dense_col = col - label_dim;
      // sizeof(int) == sizeof(float)
      const int* col_data = reinterpret_cast<const int*>(label_dense_sparse) + idx;
      dense[row * dense_dim + dense_col] = static_cast<DenseType>(dop(col_data));
    } else  // store in sparse tensors
    {
      auto col_data = label_dense_sparse[idx];  // Load column
      if constexpr (std::is_same<SparseType, long long>::value) {
        const auto sparse_col = col - label_dim - dense_dim;
        const auto bucket_id = bucket_ids[sparse_col / 2];
        const auto bucket_idx =
            (row * max_hotnesses[bucket_id] + bucket_positions[sparse_col / 2]) * 2 +
            (sparse_col & 1);
        reinterpret_cast<int**>(sparse_tensors)[bucket_id][bucket_idx] = col_data;
      } else {
        const auto sparse_col = col - label_dim - dense_dim;
        const auto bucket_id = bucket_ids[sparse_col];
        const auto bucket_idx = row * max_hotnesses[bucket_id] + bucket_positions[sparse_col];
        sparse_tensors[bucket_id][bucket_idx] = static_cast<SparseType>(col_data);
      }
    }
  }
}

template <typename DenseType, typename SparseType>
void split_3_way_feat_major(Tensor2<float> label_tensor, Tensor2<DenseType> dense_tensor,
                            Tensor2<SparseType*> sparse_tensors,
                            Tensor2<int> label_dense_sparse_tensor, Tensor2<int> bucket_ids,
                            Tensor2<int> bucket_positions, Tensor2<int> max_hotnesses,
                            cudaStream_t stream, bool dense_is_float) {
  const auto batch_size = label_dense_sparse_tensor.get_dimensions()[0];
  const auto label_dim = label_tensor.get_dimensions()[1];
  const auto dense_dim = dense_tensor.get_dimensions()[1];
  const auto sparse_dim = sparse_tensors.get_dimensions()[0];
  const auto sample_dim = label_dense_sparse_tensor.get_dimensions()[1];
  assert(label_dim > 0 && "label_dim is 0");
  assert(dense_dim > 0 && "dense_dim is 0");
  assert(sample_dim > 0 && "sample_dim is 0");

  constexpr dim3 block_dim(128);
  const dim3 grid_dim((batch_size * sample_dim + block_dim.x - 1) / block_dim.x);
  if (dense_is_float) {
    auto DOP = float_dense_op_t();
    split_feat_major_kernel<<<grid_dim, block_dim, 0, stream>>>(
        label_tensor.get_ptr(), label_dim, dense_tensor.get_ptr(), dense_dim,
        sparse_tensors.get_ptr(), sparse_dim, label_dense_sparse_tensor.get_ptr(),
        bucket_ids.get_ptr(), bucket_positions.get_ptr(), max_hotnesses.get_ptr(), batch_size,
        sample_dim, DOP);
  } else {
    auto DOP = int_dense_op_t();
    split_feat_major_kernel<<<grid_dim, block_dim, 0, stream>>>(
        label_tensor.get_ptr(), label_dim, dense_tensor.get_ptr(), dense_dim,
        sparse_tensors.get_ptr(), sparse_dim, label_dense_sparse_tensor.get_ptr(),
        bucket_ids.get_ptr(), bucket_positions.get_ptr(), max_hotnesses.get_ptr(), batch_size,
        sample_dim, DOP);
  }

  HCTR_LIB_THROW(cudaPeekAtLastError());
}

#define INSTANTIATE_SPLIT_3_WAY(DENSE_T, SPARSE_T)                                        \
  template void split_3_way_feat_major<DENSE_T, SPARSE_T>(                                \
      Tensor2<float> label_tensor, Tensor2<DENSE_T> dense_tensor,                         \
      Tensor2<SPARSE_T*> sparse_tensors, Tensor2<int> label_dense_sparse_tensor,          \
      Tensor2<int> bucket_ids, Tensor2<int> bucket_positions, Tensor2<int> max_hotnesses, \
      cudaStream_t stream, bool float_dense)

INSTANTIATE_SPLIT_3_WAY(float, uint32_t);
INSTANTIATE_SPLIT_3_WAY(__half, uint32_t);
INSTANTIATE_SPLIT_3_WAY(float, long long);
INSTANTIATE_SPLIT_3_WAY(__half, long long);

template <typename DenseType, typename SparseType>
void split_3_way_feat_major(core23::Tensor label_tensor, core23::Tensor dense_tensor,
                            core23::Tensor sparse_tensors, core23::Tensor label_dense_sparse_tensor,
                            core23::Tensor bucket_ids, core23::Tensor bucket_positions,
                            core23::Tensor max_hotnesses, cudaStream_t stream,
                            bool dense_is_float) {
  const auto batch_size = label_dense_sparse_tensor.size(0);
  const auto label_dim = label_tensor.size(1);
  const auto dense_dim = dense_tensor.size(1);
  const auto sparse_dim = sparse_tensors.size(0);
  const auto sample_dim = label_dense_sparse_tensor.size(1);
  assert(label_dim > 0 && "label_dim is 0");
  assert(dense_dim > 0 && "dense_dim is 0");
  assert(sample_dim > 0 && "sample_dim is 0");

  constexpr dim3 block_dim(128);
  const dim3 grid_dim((batch_size * sample_dim + block_dim.x - 1) / block_dim.x);
  if (dense_is_float) {
    auto DOP = float_dense_op_t();
    split_feat_major_kernel<<<grid_dim, block_dim, 0, stream>>>(
        label_tensor.data<float>(), label_dim, dense_tensor.data<DenseType>(), dense_dim,
        reinterpret_cast<SparseType**>(sparse_tensors.data()), sparse_dim,
        label_dense_sparse_tensor.data<int>(), bucket_ids.data<int>(), bucket_positions.data<int>(),
        max_hotnesses.data<int>(), batch_size, sample_dim, DOP);
  } else {
    auto DOP = int_dense_op_t();
    split_feat_major_kernel<<<grid_dim, block_dim, 0, stream>>>(
        label_tensor.data<float>(), label_dim, dense_tensor.data<DenseType>(), dense_dim,
        reinterpret_cast<SparseType**>(sparse_tensors.data()), sparse_dim,
        label_dense_sparse_tensor.data<int>(), bucket_ids.data<int>(), bucket_positions.data<int>(),
        max_hotnesses.data<int>(), batch_size, sample_dim, DOP);
  }

  HCTR_LIB_THROW(cudaPeekAtLastError());
}

#define INSTANTIATE_SPLIT_3_WAY_23(DENSE_T, SPARSE_T)                                          \
  template void split_3_way_feat_major<DENSE_T, SPARSE_T>(                                     \
      core23::Tensor label_tensor, core23::Tensor dense_tensor, core23::Tensor sparse_tensors, \
      core23::Tensor label_dense_sparse_tensor, core23::Tensor bucket_ids,                     \
      core23::Tensor bucket_positions, core23::Tensor max_hotnesses, cudaStream_t stream,      \
      bool float_dense)

INSTANTIATE_SPLIT_3_WAY_23(float, uint32_t);
INSTANTIATE_SPLIT_3_WAY_23(__half, uint32_t);
INSTANTIATE_SPLIT_3_WAY_23(float, long long);
INSTANTIATE_SPLIT_3_WAY_23(__half, long long);

}  // namespace HugeCTR

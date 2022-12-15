#include <common.hpp>
#include <data_readers/async_reader/split_label_dense_sparse.hpp>

namespace HugeCTR {

// Sparse pointer should be casted to int* when calling this kernel
template <typename DenseType, typename SparseType>
__global__ void split_kernel_3_way(int batch_size, float* label_ptr, int label_dim,
                                   DenseType* dense_ptr, int dense_dim, int dense_dim_no_align,
                                   SparseType* sparse_ptr, int sparse_dim,
                                   const int* label_dense_sparse, int sample_size_int,
                                   size_t local_idx_start, size_t local_idx_end) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < batch_size * sample_size_int) {
    const int in_col = idx % sample_size_int;
    const int in_row = idx / sample_size_int;
    const int out_row = in_row;
    if (in_col < label_dim) {
      const int out_col = in_col;
      int label = label_dense_sparse[idx];
      if (local_idx_start <= out_row && out_row < local_idx_end) {
        label_ptr[(out_row - local_idx_start) * label_dim + out_col] = label;
      }
    } else if (in_col < label_dim + dense_dim_no_align) {
      const int out_col = in_col - label_dim;
      int dense = label_dense_sparse[idx];
      if (local_idx_start <= out_row && out_row < local_idx_end) {
        dense_ptr[(out_row - local_idx_start) * dense_dim + out_col] =
            logf(dense + 1.f);  // TODO : FIXME move to data preprocessing
      }
    } else {
      const int out_col = in_col - label_dim - dense_dim_no_align;
      sparse_ptr[out_row * sparse_dim + out_col] = label_dense_sparse[idx];
    }
  }
  return;
}

template <int samples_per_cta, typename DenseType, typename SparseType>
__global__ void split_kernel_3_way_read4_write4(int batch_size, float* label_ptr, int label_dim,
                                                DenseType* dense_ptr, int dense_dim,
                                                int dense_dim_no_align, int* sparse_ptr,
                                                int sparse_dim, const int* label_dense_sparse,
                                                int sample_size_int, size_t local_idx4_start,
                                                size_t local_idx4_end) {
  using DenseType4 = typename std::conditional<(sizeof(DenseType) == 4), int4, int2>::type;
  extern __shared__ int label_dense_sparse_s[];
  constexpr int vec_size = sizeof(int4) / sizeof(int);
  static_assert(samples_per_cta % vec_size == 0,
                "Number of samples per block has to respect divisibility constraints");
  assert(blockDim.x >= 3 * warpSize);

  const int idx_l = threadIdx.x;
  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;

  const int my_cta_samples = min(samples_per_cta, batch_size - samples_per_cta * blockIdx.x);
  if (my_cta_samples <= 0) {
    return;
  }
  assert(my_cta_samples % vec_size == 0);

  int4* label_dense_sparse_s_align4 = reinterpret_cast<int4*>(label_dense_sparse_s);
  const int4* label_dense_sparse_align4 = reinterpret_cast<const int4*>(label_dense_sparse);

  float* label_s =
      reinterpret_cast<float*>(label_dense_sparse_s + sample_size_int * samples_per_cta);
  DenseType* dense_s = reinterpret_cast<DenseType*>(label_s + label_dim * samples_per_cta);
  SparseType* sparse_s = reinterpret_cast<SparseType*>((int*)dense_s + dense_dim * samples_per_cta);

  // read with int4
  const int src_base = samples_per_cta * sample_size_int / vec_size * blockIdx.x;
  for (int id = idx_l; id < my_cta_samples * sample_size_int / vec_size; id += blockDim.x) {
    label_dense_sparse_s_align4[id] = label_dense_sparse_align4[src_base + id];
  }

  for (int id = idx_l; id < samples_per_cta * dense_dim; id += blockDim.x) {
    dense_s[id] = 0;
  }

  __syncthreads();

  // transpose
  for (int id = idx_l; id < samples_per_cta * sample_size_int; id += blockDim.x) {
    const int in_col = id % sample_size_int;
    const int in_row = id / sample_size_int;
    const int out_row = in_row;
    if (in_col < label_dim) {
      const int out_col = in_col;
      label_s[out_row * label_dim + out_col] = label_dense_sparse_s[id];
    } else if (in_col < label_dim + dense_dim_no_align) {
      const int out_col = in_col - label_dim;
      int dense = label_dense_sparse_s[id];
      dense_s[out_row * dense_dim + out_col] =
          logf(dense + 1.f);  // TODO : FIXME move to data preprocessing
    } else {
      const int out_col = in_col - label_dim - dense_dim_no_align;
      sparse_s[out_row * sparse_dim + out_col] = label_dense_sparse_s[id];
    }
  }
  __syncthreads();

  float4* label_s_align4 = reinterpret_cast<float4*>(label_s);
  DenseType4* dense_s_align4 = reinterpret_cast<DenseType4*>(dense_s);
  int4* sparse_s_align4 = reinterpret_cast<int4*>(sparse_s);
  float4* label_align4 = reinterpret_cast<float4*>(label_ptr);
  DenseType4* dense_align4 = reinterpret_cast<DenseType4*>(dense_ptr);
  int4* sparse_align4 = reinterpret_cast<int4*>(sparse_ptr);

  const int label_size_int4_per_cta = label_dim * samples_per_cta / vec_size;
  const int dense_size_int4_per_cta = dense_dim * samples_per_cta / vec_size;
  const int sparse_size_int4_per_cta = sparse_dim * samples_per_cta / vec_size;

  if (warp_id == 0) {
    for (int id = lane_id; id < label_dim * my_cta_samples / vec_size; id += warpSize) {
      size_t local_idx4 = id + blockIdx.x * label_size_int4_per_cta;
      if (label_dim * local_idx4_start <= local_idx4 && local_idx4 < label_dim * local_idx4_end) {
        label_align4[local_idx4 - label_dim * local_idx4_start] = label_s_align4[id];
      }
    }
  }
  if (warp_id == 1) {
    for (int id = lane_id; id < dense_dim * my_cta_samples / vec_size; id += warpSize) {
      size_t local_idx4 = id + blockIdx.x * dense_size_int4_per_cta;
      if (dense_dim * local_idx4_start <= local_idx4 && local_idx4 < dense_dim * local_idx4_end) {
        dense_align4[local_idx4 - dense_dim * local_idx4_start] = dense_s_align4[id];
      }
    }
  }
  if (warp_id == 2) {
    for (int id = lane_id; id < sparse_dim * my_cta_samples / vec_size; id += warpSize) {
      sparse_align4[id + blockIdx.x * sparse_size_int4_per_cta] = sparse_s_align4[id];
    }
  }
}

template <typename DenseType, typename SparseType>
void split_3_way(Tensor2<float> label_tensor_per_dev, Tensor2<DenseType> dense_tensor_per_dev,
                 Tensor2<SparseType> sparse_tensor, Tensor2<int> label_dense_sparse_buffer,
                 size_t local_idx_start, size_t local_idx_end, cudaStream_t stream) {
  if (label_dense_sparse_buffer.get_num_elements() > 0) {
    assert(label_tensor_per_dev.get_dimensions()[0] == dense_tensor_per_dev.get_dimensions()[0]);
    assert(label_tensor_per_dev.get_dimensions()[0] == local_idx_end - local_idx_start);

    const int batch_size = label_dense_sparse_buffer.get_dimensions()[0];
    const int label_dim = label_tensor_per_dev.get_dimensions()[1];
    const int dense_dim = dense_tensor_per_dev.get_dimensions()[1];
    const int sparse_dim = sparse_tensor.get_dimensions()[1];
    const int sample_size_int = label_dense_sparse_buffer.get_dimensions()[1];

    int dense_dim_no_align = sample_size_int - label_dim - sparse_dim;

    constexpr int block_dim = 128;
    constexpr int samples_per_cta = 24;

    int vec_width = sizeof(int4) / sizeof(int);
    if (sizeof(SparseType) == 4 && batch_size % vec_width == 0 &&
        local_idx_start % vec_width == 0 && local_idx_end % vec_width == 0 &&
        samples_per_cta * sample_size_int * sizeof(int) <= 24 * 1024) {
      const int grid_dim = (batch_size + samples_per_cta - 1) / samples_per_cta;
      const int shmem = 2 * samples_per_cta * (label_dim + dense_dim + sparse_dim) * sizeof(int);

      split_kernel_3_way_read4_write4<samples_per_cta, DenseType, SparseType>
          <<<grid_dim, block_dim, shmem, stream>>>(
              batch_size, label_tensor_per_dev.get_ptr(), label_dim, dense_tensor_per_dev.get_ptr(),
              dense_dim, dense_dim_no_align, reinterpret_cast<int*>(sparse_tensor.get_ptr()),
              sparse_dim, label_dense_sparse_buffer.get_ptr(), sample_size_int,
              local_idx_start / vec_width, local_idx_end / vec_width);
    } else {
      const int grid_dim = (label_dense_sparse_buffer.get_num_elements() - 1) / block_dim + 1;
      split_kernel_3_way<DenseType, SparseType><<<grid_dim, block_dim, 0, stream>>>(
          batch_size, label_tensor_per_dev.get_ptr(), label_dim, dense_tensor_per_dev.get_ptr(),
          dense_dim, dense_dim_no_align, sparse_tensor.get_ptr(), sparse_dim,
          label_dense_sparse_buffer.get_ptr(), sample_size_int, local_idx_start, local_idx_end);
    }

    HCTR_LIB_THROW(cudaPeekAtLastError());
  }
}

template void split_3_way<float, uint32_t>(Tensor2<float> label_tensor_per_dev,
                                           Tensor2<float> dense_tensor_per_dev,
                                           Tensor2<uint32_t> sparse_tensor,
                                           Tensor2<int> label_dense_sparse_buffer,
                                           size_t local_idx_start, size_t local_idx_end,
                                           cudaStream_t stream);
template void split_3_way<__half, uint32_t>(Tensor2<float> label_tensor_per_dev,
                                            Tensor2<__half> dense_tensor_per_dev,
                                            Tensor2<uint32_t> sparse_tensor,
                                            Tensor2<int> label_dense_sparse_buffer,
                                            size_t local_idx_start, size_t local_idx_end,
                                            cudaStream_t stream);

template void split_3_way<float, long long>(Tensor2<float> label_tensor_per_dev,
                                            Tensor2<float> dense_tensor_per_dev,
                                            Tensor2<long long> sparse_tensor,
                                            Tensor2<int> label_dense_sparse_buffer,
                                            size_t local_idx_start, size_t local_idx_end,
                                            cudaStream_t stream);
template void split_3_way<__half, long long>(Tensor2<float> label_tensor_per_dev,
                                             Tensor2<__half> dense_tensor_per_dev,
                                             Tensor2<long long> sparse_tensor,
                                             Tensor2<int> label_dense_sparse_buffer,
                                             size_t local_idx_start, size_t local_idx_end,
                                             cudaStream_t stream);

}  // namespace HugeCTR

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

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <mma.h>

#include <common.hpp>
#include <layers/interaction_layer.hpp>
#include <type_traits>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <uint x>
struct Log2 {
  static constexpr uint value = 1 + Log2<x / 2>::value;
};
template <>
struct Log2<1> {
  static constexpr uint value = 0;
};

struct __align__(8) half4 {
  half2 vals[2];
};

template <uint WARPS_PER_BLOCK, uint THREADBLOCK_SIZE, uint M_BLOCKS, uint K_BLOCKS,
          uint SMEM_STRIDE, uint SMEM_STRIDE_ACC, uint THREADS_IN_WARP, uint THREADS_IN_WARP_LOG_2,
          uint TILE_DIM, uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractFwdKernelNonAligned(
    const __half *__restrict bottom_mlp_input, const __half *__restrict emb_input,
    __half *__restrict output, uint batch_size, uint num_rows, uint num_cols,
    uint num_rows_after_padding, uint num_cols_after_padding, uint smem_elems_per_warp,
    uint smem_rows_per_warp, uint output_size, uint num_row_steps, uint num_col_steps) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
  uint warp_id = (threadIdx.x >> THREADS_IN_WARP_LOG_2);
  int sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  int lane_id = threadIdx.x & (THREADS_IN_WARP - 1);

  extern __shared__ half shmem_dynamic[];
  half *shmem = shmem_dynamic + (warp_id * smem_elems_per_warp);

  // const half *sample_input = input + num_rows * num_cols * sample_id;
  const half *sample_bottom_mlp_input = bottom_mlp_input + num_cols * sample_id;
  const half *sample_emp_input = emb_input + (num_rows - 1) * num_cols * sample_id;
  const half *sample_input = sample_bottom_mlp_input;
  // for (uint i = 0; i < num_rows; ++i, sample_input += num_cols) {
  for (uint i = 0; i < num_rows; ++i) {
    for (uint idx = lane_id; idx < num_cols; idx += THREADS_IN_WARP) {
      (shmem + i * SMEM_STRIDE)[idx] = sample_input[idx];
    }
    sample_input = (i == 0) ? sample_emp_input : (sample_input + num_cols);
  }

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (int i = 0; i < num_rows; ++i) {
      (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
    }
  }

  half4 zeros;
  zeros.vals[0].x = __float2half(0);
  zeros.vals[0].y = __float2half(0);
  zeros.vals[1].x = __float2half(0);
  zeros.vals[1].y = __float2half(0);
  if (lane_id < (num_cols_after_padding >> 2)) {
    for (int i = num_rows; i < num_rows_after_padding; i++) {
      ((half4 *)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
    }
  }
  __syncwarp();
  half *gmem_output = output + output_size * sample_id;

  for (uint idx = lane_id; idx < num_cols; idx += THREADS_IN_WARP) {
    gmem_output[idx] = shmem[idx];
  }

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>
      acc[M_BLOCKS][M_BLOCKS];

  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      nvcuda::wmma::fill_fragment(acc[i][j], 0);
    }
  }

  for (int k_step = 0; k_step < num_col_steps; k_step++) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half,
                           nvcuda::wmma::row_major>
        a[M_BLOCKS];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half,
                           nvcuda::wmma::col_major>
        b[M_BLOCKS];
    for (int j = 0; j < M_BLOCKS; j++) {
      int base_row = (j < M_BLOCKS - 1) ? j * 16 : smem_rows_per_warp - 16;
      const half *tile_ptr = shmem + (base_row * SMEM_STRIDE + k_step * 16);
      nvcuda::wmma::load_matrix_sync(a[j], tile_ptr, SMEM_STRIDE);
      nvcuda::wmma::load_matrix_sync(b[j], tile_ptr, SMEM_STRIDE);
    }
    for (int i = 0; i < M_BLOCKS; i++) {
      for (int j = 0; j < M_BLOCKS; j++) {
        nvcuda::wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
  }
  float *shmem_store = reinterpret_cast<float *>(shmem);
  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      float *tile_ptr = shmem_store + (i * 16 * SMEM_STRIDE_ACC + j * 16);
      nvcuda::wmma::store_matrix_sync(tile_ptr, acc[i][j], SMEM_STRIDE_ACC,
                                      nvcuda::wmma::mem_row_major);
    }
  }

  half *gmem_interact_output = gmem_output + num_cols;
  int lastRowBlockOffset = M_BLOCKS * 16 - smem_rows_per_warp;
  int srcLine = 0;
  for (int i = 0; i < num_rows; ++i, ++srcLine) {
    if (i == ((M_BLOCKS - 1) * 16)) {
      srcLine += lastRowBlockOffset;
    }
    if (lane_id < i) {
      uint offset = (i * (i - 1)) >> 1;
      gmem_interact_output[offset + lane_id] =
          __float2half(shmem_store[srcLine * SMEM_STRIDE_ACC + lane_id]);
    }
  }
  // Padding
  if (lane_id == 0) {
    gmem_output[output_size - 1] = __float2half(0);
  }
#else
#warning "dotBasedInteractFwdKernelNonAligned is not supported for SM < 70 (or __CUDA_ARCH__ < 700)"
#endif
}

template <uint WARPS_PER_BLOCK, uint THREADBLOCK_SIZE, uint M_BLOCKS, uint K_BLOCKS,
          uint SMEM_STRIDE, uint SMEM_STRIDE_ACC, uint THREADS_IN_WARP, uint THREADS_IN_WARP_LOG_2,
          uint TILE_DIM, uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractFwdKernel(const __half *__restrict bottom_mlp_input,
                                   const __half *__restrict emb_input, __half *__restrict output,
                                   uint batch_size, uint num_rows, uint num_cols,
                                   uint num_rows_after_padding, uint num_cols_after_padding,
                                   uint smem_elems_per_warp, uint smem_rows_per_warp,
                                   uint output_size, uint num_row_steps, uint num_col_steps) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
  uint warp_id = (threadIdx.x >> THREADS_IN_WARP_LOG_2);
  int sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  int lane_id = threadIdx.x & (THREADS_IN_WARP - 1);

  extern __shared__ half shmem_dynamic[];
  half *shmem = shmem_dynamic + (warp_id * smem_elems_per_warp);

  // const half *sample_input = input + num_rows * num_cols * sample_id;
  const half *sample_bottom_mlp_input = bottom_mlp_input + num_cols * sample_id;
  const half *sample_emp_input = emb_input + (num_rows - 1) * num_cols * sample_id;
  const half *sample_input = sample_bottom_mlp_input;
  if (lane_id < (num_cols >> 2)) {
    // for (int i = 0; i < num_rows; ++i, sample_input += num_cols) {
    for (int i = 0; i < num_rows; ++i) {
      ((float2 *)(shmem + i * SMEM_STRIDE))[lane_id] = ((float2 *)sample_input)[lane_id];
      sample_input = (i == 0) ? sample_emp_input : (sample_input + num_cols);
    }
  }

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (int i = 0; i < num_rows; ++i) {
      (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
    }
  }

  half4 zeros;
  zeros.vals[0].x = __float2half(0);
  zeros.vals[0].y = __float2half(0);
  zeros.vals[1].x = __float2half(0);
  zeros.vals[1].y = __float2half(0);
  if (lane_id < (num_cols_after_padding >> 2)) {
    for (int i = num_rows; i < num_rows_after_padding; i++) {
      ((half4 *)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
    }
  }
  __syncwarp();
  half *gmem_output = output + output_size * sample_id;
  if (lane_id < (num_cols >> 2)) {
    ((float2 *)gmem_output)[lane_id] = ((float2 *)shmem)[lane_id];
  }

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>
      acc[M_BLOCKS][M_BLOCKS];

  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      nvcuda::wmma::fill_fragment(acc[i][j], 0);
    }
  }

  for (int k_step = 0; k_step < num_col_steps; k_step++) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half,
                           nvcuda::wmma::row_major>
        a[M_BLOCKS];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half,
                           nvcuda::wmma::col_major>
        b[M_BLOCKS];
    for (int j = 0; j < M_BLOCKS; j++) {
      int base_row = (j < M_BLOCKS - 1) ? j * 16 : smem_rows_per_warp - 16;
      const half *tile_ptr = shmem + (base_row * SMEM_STRIDE + k_step * 16);
      nvcuda::wmma::load_matrix_sync(a[j], tile_ptr, SMEM_STRIDE);
      nvcuda::wmma::load_matrix_sync(b[j], tile_ptr, SMEM_STRIDE);
    }
    for (int i = 0; i < M_BLOCKS; i++) {
      for (int j = 0; j < M_BLOCKS; j++) {
        nvcuda::wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
  }
  float *shmem_store = reinterpret_cast<float *>(shmem);
  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      float *tile_ptr = shmem_store + (i * 16 * SMEM_STRIDE_ACC + j * 16);
      nvcuda::wmma::store_matrix_sync(tile_ptr, acc[i][j], SMEM_STRIDE_ACC,
                                      nvcuda::wmma::mem_row_major);
    }
  }

  half *gmem_interact_output = gmem_output + num_cols;
  int lastRowBlockOffset = M_BLOCKS * 16 - smem_rows_per_warp;
  int srcLine = 0;
  for (int i = 0; i < num_rows; ++i, ++srcLine) {
    if (i == ((M_BLOCKS - 1) * 16)) {
      srcLine += lastRowBlockOffset;
    }
    if (lane_id < i) {
      uint offset = (i * (i - 1)) >> 1;
      gmem_interact_output[offset + lane_id] =
          __float2half(shmem_store[srcLine * SMEM_STRIDE_ACC + lane_id]);
    }
  }
  // Padding
  if (lane_id == 0) {
    gmem_output[output_size - 1] = __float2half(0);
  }
#else
#warning "dotBasedInteractFwdKernel is not supported for SM < 70 (or __CUDA_ARCH__ < 700)"
#endif
}

template <uint WARPS_PER_BLOCK, uint THREADBLOCK_SIZE, uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP, uint THREADS_IN_WARP, uint THREADS_IN_WARP_LOG_2, uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractBwdKernelNonAligned(
    const __half *__restrict upstream_grad, half __restrict *bottom_mlp_grad,
    half __restrict *emb_grad, uint batch_size, uint num_rows, uint num_cols,
    uint num_rows_after_padding, uint num_cols_after_padding, uint sample_size,
    uint interaction_ugrad_size, uint interaction_ugrad_size_with_padding,
    uint interaction_ugrad_2D_size_elems, uint interaction_ugrad_2D_stride, uint input_size_elems,
    uint input_stride, uint num_row_steps, uint num_col_steps, uint row_tiles_per_step,
    uint shared_mem_per_warp_size_byte) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
  extern __shared__ half shared_mem[];
  uint warp_id = (threadIdx.x >> THREADS_IN_WARP_LOG_2);
  uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  uint lane_id = threadIdx.x & (THREADS_IN_WARP - 1);
  // ">> 1" to convert to half pointer
  uint smem_warp_offset = warp_id * (shared_mem_per_warp_size_byte >> 1);

  half *smem_in = &shared_mem[smem_warp_offset];
  half *smem_temp = &shared_mem[smem_warp_offset + input_size_elems];
  float *smem_out = reinterpret_cast<float *>(smem_temp);

  // Global memory pointers for the current sample
  // Input
  // uint gmem_input_sample_offset = sample_id * sample_size;
  // const half *gmem_input = &input[gmem_input_sample_offset];
  uint gmem_bottom_mlp_input_sample_offset = sample_id * num_cols;
  uint gmem_emb_input_sample_offset = sample_id * (num_rows - 1) * num_cols;
  const half *gmem_bottom_mlp_input = &bottom_mlp_grad[gmem_bottom_mlp_input_sample_offset];
  const half *gmem_emb_input = &emb_grad[gmem_emb_input_sample_offset];

  // Interaction Gradient
  // const uint &gmem_grad_sample_offset = gmem_input_sample_offset;
  // half *gmem_grad = &grad[gmem_grad_sample_offset];
  half *gmem_bottom_mlp_grad = &bottom_mlp_grad[gmem_bottom_mlp_input_sample_offset];
  half *gmem_emb_grad = &emb_grad[gmem_emb_input_sample_offset];

  // Bottom MLP gradient
  // half *gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

  // Upstream gradient vector
  uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
  const half *gmem_ugrad = &upstream_grad[gmem_ugrad_sample_offset];

  // Upstream gradient vector for interactions
  const half *gmem_ugrad_interactions = &gmem_ugrad[num_cols];

// upstream grad -> shared memory (place in input section temporarily)
#pragma unroll
  for (uint idx = lane_id; idx < interaction_ugrad_size; idx += THREADS_IN_WARP) {
    smem_in[idx] = gmem_ugrad_interactions[idx];
  }
  __syncwarp();
  // Form the 2D ugrad matrix.
  if (lane_id < num_rows_after_padding) {
    uint ugrad_flat_index = ((lane_id * (lane_id - 1)) >> 1);
    uint ugrad_offset_1 = lane_id * interaction_ugrad_2D_stride;
    for (uint row = 0; row < num_rows; row++) {
      half ugrad_val = __float2half(0.0f);
      if (row < lane_id && lane_id < num_rows) {
        ugrad_val = smem_in[ugrad_flat_index + row];
        smem_temp[ugrad_offset_1 + row] = ugrad_val;
      }
      if (row <= lane_id && lane_id < num_rows_after_padding) {
        smem_temp[row * interaction_ugrad_2D_stride + lane_id] = ugrad_val;
      }
    }
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      smem_temp[row * interaction_ugrad_2D_stride + lane_id] = __float2half(0.0f);
    }
  }
  __syncwarp();

  // Input -> Shared Memory

  for (uint row = 0; row < num_rows; row++) {
    half *smem_row_ptr = &smem_in[row * input_stride];
    // const half *gmem_row_ptr = &gmem_input[row * num_cols];
    const half *gmem_row_ptr =
        (row == 0) ? gmem_bottom_mlp_input : &gmem_emb_input[(row - 1) * num_cols];
    for (uint idx = lane_id; idx < num_cols; idx += THREADS_IN_WARP) {
      smem_row_ptr[idx] = gmem_row_ptr[idx];
    }
    uint idx = lane_id + num_cols;
    if (idx < num_cols_after_padding) {
      smem_row_ptr[idx] = __float2half(0);
    }
  }

#pragma unroll 2
  for (uint row = num_rows; row < num_rows_after_padding; row++) {
    half *smem_row_ptr = &smem_in[row * input_stride];
    for (uint idx = lane_id; idx < num_cols_after_padding; idx += THREADS_IN_WARP) {
      smem_row_ptr[idx] = __float2half(0);
    }
  }
  __syncwarp();

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half,
                         nvcuda::wmma::row_major>
      a[ROW_TILES_PER_STEP][ROW_TILES_PER_STEP];
  for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
    for (uint j = 0; j < ROW_TILES_PER_STEP; j++) {
      const half *tile_ptr = smem_temp + ((i * interaction_ugrad_2D_stride + j) << TILE_DIM_LOG_2);
      nvcuda::wmma::load_matrix_sync(a[i][j], tile_ptr, interaction_ugrad_2D_stride);
    }
  }

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>
      acc[ROW_TILES_PER_STEP];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half,
                         nvcuda::wmma::row_major>
      b[ROW_TILES_PER_STEP];
  for (int col_step = 0; col_step < num_col_steps; col_step++) {
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      const half *tile_ptr = smem_in + ((i * input_stride + col_step) << TILE_DIM_LOG_2);
      nvcuda::wmma::fill_fragment(acc[i], 0);
      nvcuda::wmma::load_matrix_sync(b[i], tile_ptr, input_stride);
    }
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      for (uint j = 0; j < ROW_TILES_PER_STEP; j++) {
        nvcuda::wmma::mma_sync(acc[i], a[i][j], b[j], acc[i]);
      }
    }
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      float *tile_ptr = smem_out + i * TILE_DIM * TILE_DIM;
      nvcuda::wmma::store_matrix_sync(tile_ptr, acc[i], TILE_DIM, nvcuda::wmma::mem_row_major);
    }
    __syncwarp();
    uint gmem_grad_col = (col_step << TILE_DIM_LOG_2) + lane_id;
    if (gmem_grad_col < num_cols) {
      for (uint i = 0; i < num_rows; i++) {
        // gmem_grad[i * num_cols + gmem_grad_col] = __float2half(smem_out[(i << TILE_DIM_LOG_2) +
        // lane_id]);
        half *gmem_grad = (i == 0) ? gmem_bottom_mlp_grad : gmem_emb_grad;
        uint idx = (i == 0) ? gmem_grad_col : ((i - 1) * num_cols + gmem_grad_col);
        half val = __float2half(smem_out[(i << TILE_DIM_LOG_2) + lane_id]);
        gmem_grad[idx] = (i == 0) ? (val + gmem_ugrad[idx]) : val;
      }
    }
  }

// for (uint idx = lane_id; idx < num_cols; idx += THREADS_IN_WARP) {
//   gmem_mlp_grad[idx] = gmem_ugrad[idx];
// }
#else
#warning "dotBasedInteractBwdKernelNonAligned is not supported for SM < 70 (or __CUDA_ARCH__ < 700)"
#endif
}

template <uint WARPS_PER_BLOCK, uint THREADBLOCK_SIZE, uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP, uint THREADS_IN_WARP, uint THREADS_IN_WARP_LOG_2, uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractBwdKernel(const __half *__restrict upstream_grad,
                                   half __restrict *bottom_mlp_grad, half __restrict *emb_grad,
                                   uint batch_size, uint num_rows, uint num_cols,
                                   uint num_rows_after_padding, uint num_cols_after_padding,
                                   uint sample_size, uint interaction_ugrad_size,
                                   uint interaction_ugrad_size_with_padding,
                                   uint interaction_ugrad_2D_size_elems,
                                   uint interaction_ugrad_2D_stride, uint input_size_elems,
                                   uint input_stride, uint num_row_steps, uint num_col_steps,
                                   uint row_tiles_per_step, uint shared_mem_per_warp_size_byte) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
  extern __shared__ half shared_mem[];
  uint warp_id = (threadIdx.x >> THREADS_IN_WARP_LOG_2);
  uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  uint lane_id = threadIdx.x & (THREADS_IN_WARP - 1);
  // ">> 1" to convert to half pointer
  uint smem_warp_offset = warp_id * (shared_mem_per_warp_size_byte >> 1);

  half *smem_in = &shared_mem[smem_warp_offset];
  half *smem_temp = &shared_mem[smem_warp_offset + input_size_elems];
  float *smem_out = reinterpret_cast<float *>(smem_temp);

  // Global memory pointers for the current sample
  // Input
  // uint gmem_input_sample_offset = sample_id * sample_size;
  // const half *gmem_input = &input[gmem_input_sample_offset];
  uint gmem_bottom_mlp_input_sample_offset = sample_id * num_cols;
  uint gmem_emb_input_sample_offset = sample_id * (num_rows - 1) * num_cols;
  const half *gmem_bottom_mlp_input = &bottom_mlp_grad[gmem_bottom_mlp_input_sample_offset];
  const half *gmem_emb_input = &emb_grad[gmem_emb_input_sample_offset];

  // Interaction Gradient
  // const uint &gmem_grad_sample_offset = gmem_input_sample_offset;
  // half *gmem_grad = &grad[gmem_grad_sample_offset];
  half *gmem_bottom_mlp_grad = &bottom_mlp_grad[gmem_bottom_mlp_input_sample_offset];
  half *gmem_emb_grad = &emb_grad[gmem_emb_input_sample_offset];

  // Bottom MLP gradient
  // half *gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

  // Upstream gradient vector
  uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
  const half *gmem_ugrad = &upstream_grad[gmem_ugrad_sample_offset];

  // Upstream gradient vector for interactions
  const half *gmem_ugrad_interactions = &gmem_ugrad[num_cols];

// upstream grad -> shared memory (place in input section temporarily)
#pragma unroll
  for (uint idx = lane_id; idx < (interaction_ugrad_size >> 3); idx += THREADS_IN_WARP) {
    ((float4 *)smem_in)[idx] = ((float4 *)gmem_ugrad_interactions)[idx];
  }
  uint offset = (interaction_ugrad_size >> 3) << 3;
  for (uint idx = lane_id + offset; idx < interaction_ugrad_size; idx += THREADS_IN_WARP) {
    smem_in[idx] = gmem_ugrad_interactions[idx];
  }
  __syncwarp();
  // Form the 2D ugrad matrix.
  if (lane_id < num_rows_after_padding) {
    uint ugrad_flat_index = ((lane_id * (lane_id - 1)) >> 1);
    uint ugrad_offset_1 = lane_id * interaction_ugrad_2D_stride;
    for (uint row = 0; row < num_rows; row++) {
      half ugrad_val = __float2half(0.0f);
      if (row < lane_id && lane_id < num_rows) {
        ugrad_val = smem_in[ugrad_flat_index + row];
        smem_temp[ugrad_offset_1 + row] = ugrad_val;
      }
      if (row <= lane_id && lane_id < num_rows_after_padding) {
        smem_temp[row * interaction_ugrad_2D_stride + lane_id] = ugrad_val;
      }
    }
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      smem_temp[row * interaction_ugrad_2D_stride + lane_id] = __float2half(0.0f);
    }
  }
  __syncwarp();

  // Input -> Shared Memory

  if (lane_id < (num_cols >> 2)) {
    for (uint row = 0; row < num_rows; row++) {
      half *smem_row_ptr = &smem_in[row * input_stride];
      // const half *gmem_row_ptr = &gmem_input[row * num_cols];
      const half *gmem_row_ptr =
          (row == 0) ? gmem_bottom_mlp_input : &gmem_emb_input[(row - 1) * num_cols];
      ((float2 *)smem_row_ptr)[lane_id] = ((float2 *)gmem_row_ptr)[lane_id];
    }
  }

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (uint row = 0; row < num_rows; row++) {
      half *smem_row_ptr = &smem_in[row * input_stride];
      smem_row_ptr[idx] = __float2half(0);
    }
  }

  half4 zeros;
  zeros.vals[0].x = __float2half(0);
  zeros.vals[0].y = __float2half(0);
  zeros.vals[1].x = __float2half(0);
  zeros.vals[1].y = __float2half(0);
  if (lane_id < (num_cols_after_padding >> 2)) {
#pragma unroll 2
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      half *smem_row_ptr = &smem_in[row * input_stride];
      ((half4 *)smem_row_ptr)[lane_id] = zeros;
    }
  }
  __syncwarp();

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half,
                         nvcuda::wmma::row_major>
      a[ROW_TILES_PER_STEP][ROW_TILES_PER_STEP];
  for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
    for (uint j = 0; j < ROW_TILES_PER_STEP; j++) {
      const half *tile_ptr = smem_temp + ((i * interaction_ugrad_2D_stride + j) << TILE_DIM_LOG_2);
      nvcuda::wmma::load_matrix_sync(a[i][j], tile_ptr, interaction_ugrad_2D_stride);
    }
  }

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>
      acc[ROW_TILES_PER_STEP];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half,
                         nvcuda::wmma::row_major>
      b[ROW_TILES_PER_STEP];
  for (int col_step = 0; col_step < num_col_steps; col_step++) {
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      const half *tile_ptr = smem_in + ((i * input_stride + col_step) << TILE_DIM_LOG_2);
      nvcuda::wmma::fill_fragment(acc[i], 0);
      nvcuda::wmma::load_matrix_sync(b[i], tile_ptr, input_stride);
    }
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      for (uint j = 0; j < ROW_TILES_PER_STEP; j++) {
        nvcuda::wmma::mma_sync(acc[i], a[i][j], b[j], acc[i]);
      }
    }
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      float *tile_ptr = smem_out + i * TILE_DIM * TILE_DIM;
      nvcuda::wmma::store_matrix_sync(tile_ptr, acc[i], TILE_DIM, nvcuda::wmma::mem_row_major);
    }
    __syncwarp();
    uint gmem_grad_col_base = (col_step << TILE_DIM_LOG_2);
    uint gmem_grad_col = gmem_grad_col_base + lane_id;
    if (gmem_grad_col < num_cols) {
      if (lane_id < 8) {
        ((__half2 *)(gmem_bottom_mlp_grad + gmem_grad_col_base))[lane_id] =
            __hadd2(__float22half2_rn(((float2 *)smem_out)[lane_id]),
                    ((__half2 *)(gmem_ugrad + gmem_grad_col_base))[lane_id]);
      }
      for (uint i = 0; i < num_rows - 1; i++) {
        half val = __float2half(smem_out[((i + 1) << TILE_DIM_LOG_2) + lane_id]);
        gmem_emb_grad[i * num_cols + gmem_grad_col] = val;
      }
    }
  }
#else
#warning "dotBasedInteractBwdKernel is not supported for SM < 70 (or __CUDA_ARCH__ < 700)"
#endif
}

inline void dotBasedInteractFwd(const void *bottom_mlp_input, const void *emb_input, void *output,
                                uint batch_size, uint num_rows, uint num_cols,
                                cudaStream_t stream) {
  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kTileDim = 16;
  const uint kTileDimLog2 = Log2<kTileDim>::value;
  const uint warps_per_threadblock = 4;
  const uint threadblock_size = warps_per_threadblock * 32;
  const uint kPaddingSize = 1;
  const uint kRowTilesPerStep = 2;
  const uint kColTilesPerStep = 1;

  // num tiles
  uint num_row_tiles = (num_rows + kTileDim - 1) >> kTileDimLog2;
  uint num_col_tiles = (num_cols + kTileDim - 1) >> kTileDimLog2;

  // number of rows and columns after padding
  uint num_rows_after_padding = kTileDim << 1;
  uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

  uint num_row_steps = num_row_tiles / kRowTilesPerStep;
  uint num_col_steps = num_col_tiles / kColTilesPerStep;

  const uint K_BLOCKS = 8;
  const uint M_BLOCKS = 2;
  const uint SKEW_HALF = ((K_BLOCKS % 2) == 0) ? 8 : 0;
  const uint SMEM_STRIDE = (K_BLOCKS * 16 + SKEW_HALF);
  // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 16 to safeload a
  // tile
  const uint smem_rows_per_warp = M_BLOCKS << 4;
  const uint smem_elems_per_warp_mat = smem_rows_per_warp * SMEM_STRIDE;
  const uint SKEW_HALF_ACC = ((M_BLOCKS % 2) == 0) ? 8 : 0;
  const uint SMEM_STRIDE_ACC = (M_BLOCKS * 16 + SKEW_HALF_ACC);
  const uint smem_elems_per_warp_acc = M_BLOCKS * 16 * SMEM_STRIDE_ACC * 2;  // output in FP32
  const uint smem_elems_per_warp = (smem_elems_per_warp_mat > smem_elems_per_warp_acc)
                                       ? smem_elems_per_warp_mat
                                       : smem_elems_per_warp_acc;
  uint output_size = num_cols + (num_rows * (num_rows - 1) >> 1) + kPaddingSize;

  bool float4_predicate = !((num_cols & 7) || (output_size & 7));

  if (float4_predicate) {
    dotBasedInteractFwdKernel<warps_per_threadblock, threadblock_size, M_BLOCKS, K_BLOCKS,
                              SMEM_STRIDE, SMEM_STRIDE_ACC, kWarpSize, kWarpSizeLog2, kTileDim,
                              kTileDimLog2>
        <<<(batch_size + warps_per_threadblock - 1) / warps_per_threadblock, threadblock_size,
           warps_per_threadblock * smem_elems_per_warp * sizeof(__half), stream>>>(
            (const __half *)bottom_mlp_input, (const __half *)emb_input, (half *)output, batch_size,
            num_rows, num_cols, num_rows_after_padding, num_cols_after_padding, smem_elems_per_warp,
            smem_rows_per_warp, output_size, num_row_steps, num_col_steps);
  } else {
    dotBasedInteractFwdKernelNonAligned<warps_per_threadblock, threadblock_size, M_BLOCKS, K_BLOCKS,
                                        SMEM_STRIDE, SMEM_STRIDE_ACC, kWarpSize, kWarpSizeLog2,
                                        kTileDim, kTileDimLog2>
        <<<(batch_size + warps_per_threadblock - 1) / warps_per_threadblock, threadblock_size,
           warps_per_threadblock * smem_elems_per_warp * sizeof(__half), stream>>>(
            (const __half *)bottom_mlp_input, (const __half *)emb_input, (half *)output, batch_size,
            num_rows, num_cols, num_rows_after_padding, num_cols_after_padding, smem_elems_per_warp,
            smem_rows_per_warp, output_size, num_row_steps, num_col_steps);
  }
}

inline void dotBasedInteractBwd(void *upstream_grad, void *bottom_mlp_grad, void *emb_grad,
                                uint batch_size, uint num_rows, uint num_cols,
                                cudaStream_t stream) {
  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kTileDim = 16;
  const uint kTileDimLog2 = Log2<kTileDim>::value;
  const uint mem_skew_size = 8;
  const uint kPaddingSize = 1;
  const uint kWarpsPerBlock = 4;
  const uint kWarpsPerBlockLog2 = Log2<kWarpsPerBlock>::value;
  const uint kNumThreads = kWarpsPerBlock * kWarpSize;
  const uint kRowTilesPerStep = 2;
  const uint kColTilesPerStep = 1;

  uint row_tiles_per_step = num_rows > kTileDim ? kRowTilesPerStep : 1;

  // num tiles
  uint num_row_tiles = (num_rows + kTileDim - 1) >> kTileDimLog2;
  uint num_col_tiles = (num_cols + kTileDim - 1) >> kTileDimLog2;

  // number of rows and columns after padding
  uint num_rows_after_padding = kTileDim << 1;
  uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

  // 2D ugrad size and stride
  uint interaction_ugrad_2D_stride = num_rows_after_padding + mem_skew_size;
  uint interaction_ugrad_2D_size_elems = num_rows_after_padding * interaction_ugrad_2D_stride;
  uint interaction_ugrad_2D_size_bytes = interaction_ugrad_2D_size_elems * sizeof(half);

  // 1D ugrad size
  uint interaction_ugrad_size = num_rows * (num_rows - 1) >> 1;
  uint interaction_ugrad_size_with_padding = interaction_ugrad_size + kPaddingSize;

  // in_out place size and stride
  uint input_stride = num_cols_after_padding + mem_skew_size;
  uint input_size_elems = num_rows_after_padding * input_stride;
  uint input_size_bytes = input_size_elems * sizeof(half);

  // sample size
  uint sample_size = num_rows * num_cols;

  // output size
  uint output_size_elems = kTileDim * kTileDim * kRowTilesPerStep * kColTilesPerStep;
  uint output_size_bytes = output_size_elems * sizeof(float);

  // staging area size
  uint staging_area_size_bytes = output_size_bytes > interaction_ugrad_2D_size_bytes
                                     ? output_size_bytes
                                     : interaction_ugrad_2D_size_bytes;

  // Shared memory size
  uint shared_mem_per_warp_size_byte = input_size_bytes + staging_area_size_bytes;
  uint shared_mem_size_bytes = kWarpsPerBlock * shared_mem_per_warp_size_byte;

  uint num_blocks = (batch_size + kWarpsPerBlock - 1) >> kWarpsPerBlockLog2;
  uint num_row_steps = num_row_tiles / row_tiles_per_step;
  uint num_col_steps = num_col_tiles / kColTilesPerStep;

  bool float4_predicate = !((interaction_ugrad_size_with_padding & 7) || (num_cols & 7));
  if (float4_predicate) {
    dotBasedInteractBwdKernel<kWarpsPerBlock, kNumThreads, kRowTilesPerStep, kColTilesPerStep,
                              kWarpSize, kWarpSizeLog2, kTileDim, kTileDimLog2>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes, stream>>>(
            (const half *)upstream_grad, (half *)bottom_mlp_grad, (half *)emb_grad, batch_size,
            num_rows, num_cols, num_rows_after_padding, num_cols_after_padding, sample_size,
            interaction_ugrad_size, interaction_ugrad_size_with_padding,
            interaction_ugrad_2D_size_elems, interaction_ugrad_2D_stride, input_size_elems,
            input_stride, num_row_steps, num_col_steps, row_tiles_per_step,
            shared_mem_per_warp_size_byte);
  } else {
    dotBasedInteractBwdKernelNonAligned<kWarpsPerBlock, kNumThreads, kRowTilesPerStep,
                                        kColTilesPerStep, kWarpSize, kWarpSizeLog2, kTileDim,
                                        kTileDimLog2>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes, stream>>>(
            (const half *)upstream_grad, (half *)bottom_mlp_grad, (half *)emb_grad, batch_size,
            num_rows, num_cols, num_rows_after_padding, num_cols_after_padding, sample_size,
            interaction_ugrad_size, interaction_ugrad_size_with_padding,
            interaction_ugrad_2D_size_elems, interaction_ugrad_2D_stride, input_size_elems,
            input_stride, num_row_steps, num_col_steps, row_tiles_per_step,
            shared_mem_per_warp_size_byte);
  }
}

template <typename T>
__global__ void concat_kernel(bool forward, T *out, T *in_mlp, T *in_emb, const int h,
                              const int out_w, const int in_w, const int n_emb) {
  const int n_ins = 1 + n_emb;
  if (blockIdx.x < n_ins) {
    T *in = (blockIdx.x == 0) ? in_mlp : in_emb + (blockIdx.x - 1) * in_w;
    for (int bid = blockIdx.y; bid < h; bid += gridDim.y) {
      int in_idx_base = (blockIdx.x == 0) ? bid * in_w : bid * in_w * n_emb;
      for (int tid = threadIdx.x; tid < in_w; tid += blockDim.x) {
        int in_idx = in_idx_base + tid;
        int out_idx = bid * out_w + blockIdx.x * in_w + tid;
        if (forward) {
          out[out_idx] = in[in_idx];
        } else {
          in[in_idx] = (blockIdx.x == 0) ? (in[in_idx] + out[out_idx]) : out[out_idx];
        }
      }
    }
  }
}

template <typename T>
__global__ void gather_concat_fprop_kernel(T *out, const T *in0, const T *mat, const int h,
                                           const int n_ins, const int w) {
  extern __shared__ T s_buf[];
  for (int bid = blockIdx.x; bid < h; bid += gridDim.x) {
    int g_in_idx_base = bid * n_ins * n_ins;
    for (int row = threadIdx.y; row < n_ins; row += blockDim.y) {
      for (int col = threadIdx.x; col < n_ins; col += blockDim.x) {
        if (col > row) {
          int idx_in_blk = row * n_ins + col;
          int g_in_idx = g_in_idx_base + idx_in_blk;
          int s_idx = (col * (col - 1) / 2) + row;
          s_buf[s_idx] = mat[g_in_idx];
        }
      }
    }
    __syncthreads();
    int tid_base = threadIdx.y * blockDim.x + threadIdx.x;
    int out_len = w + (n_ins * (n_ins + 1) / 2 - n_ins) + 1;
    int g_out_idx_base = bid * out_len;
    for (int tid = tid_base; tid < out_len - 1; tid += blockDim.y * blockDim.x) {
      int g_out_idx = g_out_idx_base + tid;
      T value = (tid < w) ? in0[bid * w + tid] : s_buf[tid - w];
      out[g_out_idx] = value;
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void transpose_and_add(const T *src, T *dst, const int h, const int n_ins) {
  extern __shared__ T s_buf[];
  for (int bid = blockIdx.z; bid < h; bid += gridDim.z) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = bid * n_ins * n_ins + y * n_ins + x;
    int sid_n = threadIdx.y * blockDim.x + threadIdx.x;
    int sid_t = threadIdx.x * blockDim.y + threadIdx.y;
    if (x < n_ins && y < n_ins) {
      s_buf[sid_n] = src[gid];
    }
    __syncthreads();
    if (x < n_ins && y < n_ins) {
      dst[gid] = s_buf[sid_n] + s_buf[sid_t];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void gather_concat_bprop_kernel(const T *out, T *in0, T *mat, const int h,
                                           const int n_ins, const int w) {
  extern __shared__ T s_buf[];
  for (int bid = blockIdx.x; bid < h; bid += gridDim.x) {
    int tid_base = threadIdx.y * blockDim.x + threadIdx.x;
    int out_len = w + (n_ins * (n_ins + 1) / 2 - n_ins) + 1;
    int g_out_idx_base = bid * out_len;
    for (int tid = tid_base; tid < out_len - 1; tid += blockDim.y * blockDim.x) {
      int g_out_idx = g_out_idx_base + tid;
      T val = out[g_out_idx];
      if (tid < w) {
        in0[bid * w + tid] = val;
      } else {
        s_buf[tid - w] = val;
      }
    }
    __syncthreads();

    int g_in_idx_base = bid * n_ins * n_ins;
    for (int row = threadIdx.y; row < n_ins; row += blockDim.y) {
      for (int col = threadIdx.x; col < n_ins; col += blockDim.x) {
        int idx_in_blk = row * n_ins + col;
        int g_in_idx = g_in_idx_base + idx_in_blk;
        int s_idx = (col * (col - 1) / 2) + row;
        mat[g_in_idx] = (col > row) ? s_buf[s_idx] : T(0);
      }
    }
    __syncthreads();
  }
}

}  // anonymous namespace

template <typename T>
InteractionLayer<T>::InteractionLayer(
    const Tensor2<T> &in_bottom_mlp_tensor, const Tensor2<T> &in_embeddings, Tensor2<T> &out_tensor,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &blobs_buff,
    const std::shared_ptr<GPUResource> &gpu_resource, bool use_mixed_precision,
    bool enable_tf32_compute)
    : Layer(gpu_resource),
      use_mixed_precision_(use_mixed_precision),
      enable_tf32_compute_(enable_tf32_compute) {
  try {
    auto first_in_dims = in_bottom_mlp_tensor.get_dimensions();
    auto second_in_dims = in_embeddings.get_dimensions();

    if (first_in_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "Input Bottom MLP must be a 2D tensor");
    }

    if (second_in_dims.size() != 3) {
      CK_THROW_(Error_t::WrongInput, "Input Embeddings must be a 3D tensor");
    }

    if (first_in_dims[0] != second_in_dims[0]) {
      CK_THROW_(Error_t::WrongInput, "the input tensors' batch sizes must be the same");
    }

    if (first_in_dims[1] != second_in_dims[2]) {
      CK_THROW_(Error_t::WrongInput, "the input tensors' widths must be the same");
    }

    size_t n_ins = 1 + second_in_dims[1];
    if (std::is_same<T, __half>::value == false) {
      size_t concat_dims_width = first_in_dims[1] + second_in_dims[1] * second_in_dims[2];
      std::vector<size_t> concat_dims = {first_in_dims[0], concat_dims_width};

      {
        Tensor2<T> tensor;
        blobs_buff->reserve(concat_dims, &tensor);
        internal_tensors_.push_back(tensor);
      }
      {
        std::vector<size_t> mat_dims = {first_in_dims[0], n_ins * n_ins};
        Tensor2<T> tensor;
        blobs_buff->reserve(mat_dims, &tensor);
        internal_tensors_.push_back(tensor);
      }
      {
        Tensor2<T> tensor;
        blobs_buff->reserve(concat_dims, &tensor);
        internal_tensors_.push_back(tensor);
      }
    }

    int concat_len = n_ins * (n_ins + 1) / 2 - n_ins;
    std::vector<size_t> out_dims = {first_in_dims[0], first_in_dims[1] + concat_len + 1};
    blobs_buff->reserve(out_dims, &out_tensor);

    in_tensors_.push_back(in_bottom_mlp_tensor);
    in_tensors_.push_back(in_embeddings);
    out_tensors_.push_back(out_tensor);

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
InteractionLayer<T>::~InteractionLayer(){};

template <typename T>
void InteractionLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  // phase 0: concat
  T *concat = internal_tensors_[0].get_ptr();
  T *in_mlp = get_in_tensors(is_train)[0].get_ptr();
  T *in_emb = get_in_tensors(is_train)[1].get_ptr();
  const int h = internal_tensors_[0].get_dimensions()[0];
  const int out_w = internal_tensors_[0].get_dimensions()[1];
  const int in_w = get_in_tensors(is_train)[0].get_dimensions()[1];
  const int n_emb = get_in_tensors(is_train)[1].get_dimensions()[1];
  const int n_ins = 1 + n_emb;

  dim3 grid0(n_ins, get_gpu().get_sm_count(), 1);
  dim3 block0(((in_w <= 128) ? 128 : ((in_w <= 256) ? 256 : 512)), 1, 1);
  concat_kernel<<<grid0, block0, 0, get_gpu().get_stream()>>>(true, concat, in_mlp, in_emb, h,
                                                              out_w, in_w, n_emb);

  // phase 1: matmul
  const int batch_count = h;
  T *mat = internal_tensors_[1].get_ptr();
  const int m = n_ins;
  const int n = n_ins;
  const int k = in_w;
  float alpha = 1.0f;
  float beta = 0.0f;
  long long int stride_a = n * k;
  long long int stride_b = k * m;
  long long int stride_c = n * m;
  cudaDataType_t a_type = CUDA_R_32F;
  cudaDataType_t b_type = CUDA_R_32F;
  cudaDataType_t c_type = CUDA_R_32F;

  cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  cublasGemmAlgo_t algo =
      use_mixed_precision_ ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;

  CK_CUBLAS_THROW_(
      cublasGemmStridedBatchedEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                 &alpha, concat, a_type, k, stride_a, concat, b_type, k, stride_b,
                                 &beta, mat, c_type, n, stride_c, batch_count, compute_type, algo));

  // phase 2: gather & concat
  T *in0 = get_in_tensors(is_train)[0].get_ptr();
  T *gather = out_tensors_[0].get_ptr();

  dim3 grid1(get_gpu().get_sm_count() * 8, 1, 1);
  dim3 block1(16, 16, 1);
  size_t smem_size = sizeof(T) * (n_ins * (n_ins + 1) / 2 - n_ins);
  gather_concat_fprop_kernel<<<grid1, block1, smem_size, get_gpu().get_stream()>>>(gather, in0, mat,
                                                                                   h, n_ins, in_w);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <>
void InteractionLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  __half *in_mlp = get_in_tensors(is_train)[0].get_ptr();
  __half *in_emb = get_in_tensors(is_train)[1].get_ptr();
  __half *output = out_tensors_[0].get_ptr();
  const int h = get_in_tensors(is_train)[0].get_dimensions()[0];
  const int in_w = get_in_tensors(is_train)[0].get_dimensions()[1];
  const int n_emb = get_in_tensors(is_train)[1].get_dimensions()[1];
  const int n_ins = 1 + n_emb;

  dotBasedInteractFwd(in_mlp, in_emb, output, h, n_ins, in_w, get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
void InteractionLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  // phase 0:
  T *gather = out_tensors_[0].get_ptr();
  T *in0 = get_in_tensors(true)[0].get_ptr();
  T *mat = internal_tensors_[1].get_ptr();
  const int h = internal_tensors_[0].get_dimensions()[0];
  const int n_ins = 1 + get_in_tensors(true)[1].get_dimensions()[1];
  const int in_w = get_in_tensors(true)[0].get_dimensions()[1];

  dim3 grid1(get_gpu().get_sm_count() * 8, 1, 1);
  dim3 block1(16, 16, 1);
  size_t smem_size = sizeof(T) * (n_ins * (n_ins + 1) / 2 - n_ins);
  gather_concat_bprop_kernel<<<grid1, block1, smem_size, get_gpu().get_stream()>>>(gather, in0, mat,
                                                                                   h, n_ins, in_w);

  // phase 1:
  const int batch_count = h;
  T *concat = internal_tensors_[0].get_ptr();
  T *concat_tmp = internal_tensors_[2].get_ptr();
  const int m = n_ins;
  const int n = in_w;
  const int k = n_ins;
  T alpha = 1.0f;
  T beta = 0.0f;
  long long int stride_a = n * k;
  long long int stride_b = k * m;
  long long int stride_c = n * m;
  cudaDataType_t a_type = CUDA_R_32F;
  cudaDataType_t b_type = CUDA_R_32F;
  cudaDataType_t c_type = CUDA_R_32F;

  cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  cublasGemmAlgo_t algo =
      use_mixed_precision_ ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;

  // mat = mat + T(mat)
  {
    dim3 block(32, 32, 1);
    dim3 grid((n_ins + block.x - 1) / block.x, (n_ins + block.y - 1) / block.y, h);
    size_t smem_size = sizeof(T) * block.x * block.y;
    transpose_and_add<<<grid, block, smem_size, get_gpu().get_stream()>>>(mat, mat, h, n_ins);
  }

  CK_CUBLAS_THROW_(cublasGemmStridedBatchedEx(
      get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, concat, a_type, n,
      stride_a, mat, b_type, k, stride_b, &beta, concat_tmp, c_type, n, stride_c, batch_count,
      compute_type, algo));

  // phase 2:
  T *in_mlp = get_in_tensors(true)[0].get_ptr();
  T *in_emb = get_in_tensors(true)[1].get_ptr();
  const int out_w = internal_tensors_[0].get_dimensions()[1];
  const int n_emb = get_in_tensors(true)[1].get_dimensions()[1];

  dim3 grid0(n_ins, get_gpu().get_sm_count(), 1);
  dim3 block0(((in_w <= 128) ? 128 : ((in_w <= 256) ? 256 : 512)), 1, 1);
  concat_kernel<<<grid0, block0, 0, get_gpu().get_stream()>>>(false, concat_tmp, in_mlp, in_emb, h,
                                                              out_w, in_w, n_emb);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <>
void InteractionLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());

  __half *up_grad = out_tensors_[0].get_ptr();
  __half *mlp_grad = get_in_tensors(true)[0].get_ptr();
  __half *emb_grad = get_in_tensors(true)[1].get_ptr();
  const int h = get_in_tensors(true)[0].get_dimensions()[0];
  const int n_emb = get_in_tensors(true)[1].get_dimensions()[1];
  const int n_ins = 1 + n_emb;
  const int in_w = get_in_tensors(true)[0].get_dimensions()[1];

  dotBasedInteractBwd(up_grad, mlp_grad, emb_grad, h, n_ins, in_w, get_gpu().get_stream());
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class InteractionLayer<float>;
template class InteractionLayer<__half>;

}  // namespace HugeCTR

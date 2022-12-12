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

#include <cublas_v2.h>
#include <linalg/gemv.h>
#include <math.h>

#include <cuda/std/array>
#include <layers/multi_cross_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/matrix_vector_op.cuh>
#include <linalg/reduce.cuh>
#include <prims/cuda_utils.cuh>
#include <prims/linalg/matrix_multiplication.cuh>
#include <utils.cuh>
#include <utils.hpp>
#include <vector>

/** Overload of built-in atomicAdd for support on Pascal architectures */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ < 700

__inline__ __device__ __half atomicAdd(__half* address, __half val) {
  size_t base_offset = ((size_t)address & 2);
  uint32_t* base_address = (uint32_t*)((char*)(address)-base_offset);

  uint32_t old = *base_address, assumed;
  do {
    assumed = old;
    {
      __half assumed_f16 = __ushort_as_half((uint16_t)(assumed >> (base_offset << 3)));
      uint32_t new_val = assumed;
      ((uint16_t*)(&new_val))[base_offset >> 1] = __half_as_ushort(__hadd(assumed_f16, val));
      old = atomicCAS(base_address, assumed, new_val);
    }
  } while (assumed != old);
  return __ushort_as_half((uint16_t)(old >> (base_offset << 3)));
}

#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ < 700

namespace HugeCTR {
struct alignas(8) half2x4 : public cuda::std::array<__half2, 4> {};
// kernels
namespace {

inline int calc_grid(int t, int b) { return (t - 1) / b + 1; }
template <typename T>
void matrix_vec_mul(Tensor2<T>& out, const Tensor2<T>& mat, const Tensor2<T>& vec,
                    cublasHandle_t cublas_handle, cudaStream_t stream);

template <>
void matrix_vec_mul(Tensor2<float>& out, const Tensor2<float>& mat, const Tensor2<float>& vec,
                    cublasHandle_t cublas_handle, cudaStream_t stream) {
  float* pout = out.get_ptr();
  const float* pmat = mat.get_ptr();
  const float* pvec = vec.get_ptr();

  const auto& dim = out.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2 && idim[1] == vec.get_dimensions()[1] &&
         vec.get_dimensions()[0] == 1);
  assert(idim[0] == dim[0]);

  const int h = idim[0];
  const int w = idim[1];
  const float alpha = 1.0f;
  const float beta = 0.0f;

  CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, h, 1, w, &alpha, pmat, w, pvec,
                           w, &beta, pout, h));
}

template <>
void matrix_vec_mul(Tensor2<__half>& out, const Tensor2<__half>& mat, const Tensor2<__half>& vec,
                    cublasHandle_t cublas_handle, cudaStream_t stream) {
  __half* pout = out.get_ptr();
  const __half* pmat = mat.get_ptr();
  const __half* pvec = vec.get_ptr();

  const auto& dim = out.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2 && idim[1] == vec.get_dimensions()[1] &&
         vec.get_dimensions()[0] == 1);
  assert(idim[0] == dim[0]);

  const int h = idim[0];
  const int w = idim[1];
  const __half alpha = 1.0f;
  const __half beta = 0.0f;

  CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  CUBLAS_CHECK(cublasHgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, h, 1, w, &alpha, pmat, w, pvec,
                           w, &beta, pout, h));
}

template <typename T>
void row_scaling(Tensor2<T>& o_mat, const Tensor2<T>& mat, const Tensor2<T>& vec,
                 cudaStream_t stream) {
  T* pout = o_mat.get_ptr();
  const T* pmat = mat.get_ptr();
  const T* pvec = vec.get_ptr();

  const auto& dim = o_mat.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2 && dim[0] == vec.get_dimensions()[0] &&
         vec.get_dimensions()[1] == 1);
  assert(idim[0] == dim[0] && idim[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  MLCommon::LinAlg::matrixVectorOp(
      pout, pmat, pvec, h, w, false, true, [] __device__(T a, T b) { return a * b; }, stream);
}

template <typename T>
void matrix_vec_add(Tensor2<T>& o_mat, const Tensor2<T>& mat, const Tensor2<T>& vec,
                    cudaStream_t stream) {
  T* pout = o_mat.get_ptr();
  const T* pmat = mat.get_ptr();
  const T* pvec = vec.get_ptr();

  const auto& dim = o_mat.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2 && dim[1] == vec.get_dimensions()[1] &&
         vec.get_dimensions()[0] == 1);
  assert(idim[0] == dim[0] && idim[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  MLCommon::LinAlg::matrixVectorOp(
      pout, pmat, pvec, h, w, false, false, [] __device__(T a, T b) { return a + b; }, stream);
}

template <typename T>
void matrix_add(Tensor2<T>& out_mat, const Tensor2<T>& mat_a, const Tensor2<T>& mat_b,
                cudaStream_t stream) {
  T* pout = out_mat.get_ptr();
  const T* pmat_a = mat_a.get_ptr();
  const T* pmat_b = mat_b.get_ptr();

  const auto& dim = out_mat.get_dimensions();
  const auto& idim1 = mat_a.get_dimensions();
  const auto& idim2 = mat_b.get_dimensions();
  assert(idim1[0] == dim[0] && idim1[1] == dim[1]);
  assert(idim2[0] == dim[0] && idim2[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  MLCommon::LinAlg::binaryOp(
      pout, pmat_a, pmat_b, h * w, [] __device__(T a, T b) { return a + b; }, stream);
}

template <typename T>
__global__ void vector_fma4(T* pout, const T* pvec_a, const T* pvec_b, const T* pvec_c,
                            const int len) {
  const int gtid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gtid < len) pout[gtid] = pvec_a[gtid] * pvec_b[gtid] + pvec_c[gtid];
}
// TODO: unrolling
template <typename T>
__global__ void vector_fma4_align8(T* pout, const T* pvec_a, const T* pvec_b, const T* pvec_c,
                                   const int len) {
  const int gtid = (blockDim.x * blockIdx.x + threadIdx.x) << 3;
  if (gtid >= len) {
    return;
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    pout[gtid + i] = pvec_a[gtid + i] * pvec_b[gtid + i] + pvec_c[gtid + i];
  }
}

// TODO: unrolling
template <typename T>
__global__ void vector_fma3_align8(T* __restrict__ pout, const T* __restrict__ pvec_a,
                                   const T* __restrict__ pvec_b, const int len) {
  const int gtid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gtid < len) pout[gtid] += pvec_a[gtid] * pvec_b[gtid];
}
// out0 = a * b
// out1 += a * c
template <typename T, int VecLen = 1, int SHT = 0>
__global__ void vector_mul_fma3_align(T* __restrict__ pout0, T* __restrict__ pout1,
                                      const T* __restrict__ pvec_a, const T* __restrict__ pvec_b,
                                      const T* __restrict__ pvec_c, const int len) {
  const int gtid = (blockDim.x * blockIdx.x + threadIdx.x) << SHT;
  if (gtid >= len) {
    return;
  }
  const T* pA = pvec_a + gtid;
  const T* pB = pvec_b + gtid;
  const T* pC = pvec_c + gtid;
  T* out1 = pout1 + gtid;
  T* out0 = pout0 + gtid;
  T regA[VecLen], regB[VecLen], regC[VecLen];
  T mul[VecLen];
  T acc[VecLen];

// load
#pragma unroll
  for (int i = 0; i < VecLen; i++) {
    regA[i] = pA[i];
    regB[i] = pB[i];
    regC[i] = pC[i];
    acc[i] = out1[i];
  }
// mul & fma
#pragma unroll
  for (int i = 0; i < VecLen; i++) {
    mul[i] = regA[i] * regB[i];
    acc[i] += regA[i] * regC[i];
  }
// store
#pragma unroll
  for (int i = 0; i < VecLen; i++) {
    out0[i] = mul[i];
    out1[i] = acc[i];
  }
}
// out0 = a * b
// out1 += a * c
template <>
__global__ void vector_mul_fma3_align<__half, 8, 3>(
    __half* __restrict__ pout0, __half* __restrict__ pout1, const __half* __restrict__ pvec_a,
    const __half* __restrict__ pvec_b, const __half* __restrict__ pvec_c, const int len) {
  const int gtid = (blockDim.x * blockIdx.x + threadIdx.x) << 3;
  if (gtid >= len) {
    return;
  }
  float4 a_8, b_8, c_8, acc_8;
  half2x4 out0, out1;
  half2x4 *out0_ptr, *out1_ptr;
  out0_ptr = reinterpret_cast<half2x4*>(pout0 + gtid);
  out1_ptr = reinterpret_cast<half2x4*>(pout1 + gtid);
  // load
  a_8 = *reinterpret_cast<const float4*>(pvec_a + gtid);
  b_8 = *reinterpret_cast<const float4*>(pvec_b + gtid);
  c_8 = *reinterpret_cast<const float4*>(pvec_c + gtid);
  acc_8 = *reinterpret_cast<const float4*>(pout1 + gtid);
  // mul
  out0[0] = __hmul2(*reinterpret_cast<half2*>(&a_8.x), *reinterpret_cast<half2*>(&b_8.x));
  out0[1] = __hmul2(*reinterpret_cast<half2*>(&a_8.y), *reinterpret_cast<half2*>(&b_8.y));
  out0[2] = __hmul2(*reinterpret_cast<half2*>(&a_8.z), *reinterpret_cast<half2*>(&b_8.z));
  out0[3] = __hmul2(*reinterpret_cast<half2*>(&a_8.w), *reinterpret_cast<half2*>(&b_8.w));
  *out0_ptr = out0;
  // add
  out1[0] = __hfma2(*reinterpret_cast<half2*>(&a_8.x), *reinterpret_cast<half2*>(&c_8.x),
                    *reinterpret_cast<half2*>(&acc_8.x));
  out1[1] = __hfma2(*reinterpret_cast<half2*>(&a_8.y), *reinterpret_cast<half2*>(&c_8.y),
                    *reinterpret_cast<half2*>(&acc_8.y));
  out1[2] = __hfma2(*reinterpret_cast<half2*>(&a_8.z), *reinterpret_cast<half2*>(&c_8.z),
                    *reinterpret_cast<half2*>(&acc_8.z));
  out1[3] = __hfma2(*reinterpret_cast<half2*>(&a_8.w), *reinterpret_cast<half2*>(&c_8.w),
                    *reinterpret_cast<half2*>(&acc_8.w));
  // store
  *out1_ptr = out1;
}
// d = a * b + c
template <>
__global__ void vector_fma4_align8(__half* pout, const __half* pvec_a, const __half* pvec_b,
                                   const __half* pvec_c, const int len) {
  const int gtid = (blockDim.x * blockIdx.x + threadIdx.x) << 3;
  if (gtid >= len) {
    return;
  }
  float4 a_8, b_8, c_8;
  half2x4 d_8;
  half2x4* out_ptr;
  out_ptr = reinterpret_cast<half2x4*>(pout + gtid);
  // load
  a_8 = *reinterpret_cast<const float4*>(pvec_a + gtid);
  b_8 = *reinterpret_cast<const float4*>(pvec_b + gtid);
  c_8 = *reinterpret_cast<const float4*>(pvec_c + gtid);
  // fma
  d_8[0] = __hfma2(*reinterpret_cast<half2*>(&a_8.x), *reinterpret_cast<half2*>(&b_8.x),
                   *reinterpret_cast<half2*>(&c_8.x));
  d_8[1] = __hfma2(*reinterpret_cast<half2*>(&a_8.y), *reinterpret_cast<half2*>(&b_8.y),
                   *reinterpret_cast<half2*>(&c_8.y));
  d_8[2] = __hfma2(*reinterpret_cast<half2*>(&a_8.z), *reinterpret_cast<half2*>(&b_8.z),
                   *reinterpret_cast<half2*>(&c_8.z));
  d_8[3] = __hfma2(*reinterpret_cast<half2*>(&a_8.w), *reinterpret_cast<half2*>(&b_8.w),
                   *reinterpret_cast<half2*>(&c_8.w));
  // store
  *out_ptr = d_8;
}
// c = a * b + c
template <>
__global__ void vector_fma3_align8(__half* __restrict__ pout, const __half* __restrict__ pvec_a,
                                   const __half* __restrict__ pvec_b, const int len) {
  const int gtid = (blockDim.x * blockIdx.x + threadIdx.x) << 3;
  if (gtid >= len) {
    return;
  }
  float4 a_8, b_8, c_8;
  half2x4 d_8;
  half2x4* out_ptr;
  out_ptr = reinterpret_cast<half2x4*>(pout + gtid);
  // load
  a_8 = *reinterpret_cast<const float4*>(pvec_a + gtid);
  b_8 = *reinterpret_cast<const float4*>(pvec_b + gtid);
  c_8 = *reinterpret_cast<const float4*>(pout + gtid);
  // fma
  d_8[0] = __hfma2(*reinterpret_cast<half2*>(&a_8.x), *reinterpret_cast<half2*>(&b_8.x),
                   *reinterpret_cast<half2*>(&c_8.x));
  d_8[1] = __hfma2(*reinterpret_cast<half2*>(&a_8.y), *reinterpret_cast<half2*>(&b_8.y),
                   *reinterpret_cast<half2*>(&c_8.y));
  d_8[2] = __hfma2(*reinterpret_cast<half2*>(&a_8.z), *reinterpret_cast<half2*>(&b_8.z),
                   *reinterpret_cast<half2*>(&c_8.z));
  d_8[3] = __hfma2(*reinterpret_cast<half2*>(&a_8.w), *reinterpret_cast<half2*>(&b_8.w),
                   *reinterpret_cast<half2*>(&c_8.w));
  // store
  *out_ptr = d_8;
}

// Y0 = A .* B
// Y1 += A .* C
template <typename T>
void fused_mul_fma3(Tensor2<T>& Y0, Tensor2<T>& Y1, const Tensor2<T>& A, const Tensor2<T>& B,
                    const Tensor2<T>& C, cudaStream_t stream) {
  const T* pmat_a = A.get_ptr();
  const T* pmat_b = B.get_ptr();
  const T* pmat_c = C.get_ptr();
  T* pmat_o0 = Y0.get_ptr();
  T* pmat_o1 = Y1.get_ptr();
  const auto& idima = A.get_dimensions();
  const auto& idimb = B.get_dimensions();
  const auto& idimc = C.get_dimensions();
  const auto& idimc0 = Y0.get_dimensions();
  const auto& idimc1 = Y1.get_dimensions();

  assert(idima[0] == idimb[0] && idima[1] == idimb[1] && idimc0[0] == idimb[0] &&
         idimc0[1] == idimb[1] && idimc[0] == idima[0] && idimc[1] == idima[1]);
  assert(idimc1[0] == idimc0[0] && idimc1[1] == idimc0[1]);
  const int h = idima[0];
  const int w = idima[1];
  const int len = h * w;
  constexpr int warp_per_sm = 8;
  constexpr int warp_size = 32;
  const int BLOCK_DIM = warp_size * warp_per_sm;  // 8 warps per block
  int GRID_DIM = (len + BLOCK_DIM - 1) / BLOCK_DIM;
  if (len % 8 == 0 && std::is_same<T, __half>::value) {
    GRID_DIM = (len / 8 + BLOCK_DIM - 1) / BLOCK_DIM;
    vector_mul_fma3_align<T, 8, 3>
        <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pmat_o0, pmat_o1, pmat_a, pmat_b, pmat_c, len);
  } else {
    vector_mul_fma3_align<T>
        <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pmat_o0, pmat_o1, pmat_a, pmat_b, pmat_c, len);
  }
}
// perform out_mat = mat_a * mat_b + mat_c
template <typename T>
void fused_matrix_elementwise_dot_add(Tensor2<T>& out_mat, const Tensor2<T>& mat_a,
                                      const Tensor2<T>& mat_b, const Tensor2<T>& mat_c,
                                      cudaStream_t stream) {
  T* pout = out_mat.get_ptr();
  const T* pmat_a = mat_a.get_ptr();
  const T* pmat_b = mat_b.get_ptr();
  const T* pmat_c = mat_c.get_ptr();
  const auto& dim = out_mat.get_dimensions();
  const auto& idima = mat_a.get_dimensions();
  const auto& idimb = mat_b.get_dimensions();
  const auto& idimc = mat_c.get_dimensions();
  assert(idima[0] == dim[0] && idima[1] == dim[1] && idimc[0] == dim[0]);
  assert(idimb[0] == dim[0] && idimb[1] == dim[1] && idimc[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  constexpr int sm_count = 108;
  constexpr int warp_per_sm = 8;
  constexpr int warp_size = 32;
  constexpr int kNumWaves = 32;
  const int BLOCK_DIM = warp_size * warp_per_sm;  // 8 warps per block
  const int GRID_DIM = (h * w + BLOCK_DIM - 1) / BLOCK_DIM;
  if (h * w % 8 == 0 && std::is_same<T, __half>::value) {
    int num_items = h * w / 8;
    const int GRID_DIM_h4 = (num_items + BLOCK_DIM - 1) / BLOCK_DIM;
    if (pout == pmat_c) {
      vector_fma3_align8<<<GRID_DIM_h4, BLOCK_DIM, 0, stream>>>(pout, pmat_a, pmat_b, h * w);
    } else {
      vector_fma4_align8<<<GRID_DIM_h4, BLOCK_DIM, 0, stream>>>(pout, pmat_a, pmat_b, pmat_c,
                                                                h * w);
    }
  } else {
    vector_fma4<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat_a, pmat_b, pmat_c, h * w);
  }
}
// c = a * b => 3
template <typename T>
void matrix_elementwise_dot(Tensor2<T>& out_mat, const Tensor2<T>& mat_a, const Tensor2<T>& mat_b,
                            cudaStream_t stream) {
  T* pout = out_mat.get_ptr();
  const T* pmat_a = mat_a.get_ptr();
  const T* pmat_b = mat_b.get_ptr();

  const auto& dim = out_mat.get_dimensions();
  const auto& idim1 = mat_a.get_dimensions();
  const auto& idim2 = mat_b.get_dimensions();
  assert(idim1[0] == dim[0] && idim1[1] == dim[1]);
  assert(idim2[0] == dim[0] && idim2[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  MLCommon::LinAlg::binaryOp(
      pout, pmat_a, pmat_b, h * w, [] __device__(T a, T b) { return a * b; }, stream);
}

/**
 * compute dot product for each pair of the rows in the two matrix,
 */
template <typename T>
__global__ void matrix_pair_mul_kernel(T* o_vec, const T* mat_a, int h, int w, const T* mat_b) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int wtid = tid % WARP_SIZE;  // thread id in warp
  const int wid = tid / WARP_SIZE;   // warp id
  const T* mat_a_with_offset = mat_a + wid * w;
  const T* mat_b_with_offset = mat_b + wid * w;
  if (wid < h) {
    T accum = 0.f;
    for (int i = wtid; i < w; i += WARP_SIZE) {
      accum += mat_a_with_offset[i] * mat_b_with_offset[i];
    }
    T val = warpReduceSum(accum);
    if (wtid == 0) {
      o_vec[wid] = val;
    }
  }
}

template <typename T>
void matrix_pair_mul(Tensor2<T>& o_vec, const Tensor2<T>& mat_a, const Tensor2<T>& mat_b,
                     cudaStream_t stream) {
  T* pout = o_vec.get_ptr();
  const T* pmat_a = mat_a.get_ptr();
  const T* pmat_b = mat_b.get_ptr();

  const auto& dim = mat_a.get_dimensions();

  const int h = dim[0];
  const int w = dim[1];
  assert(h == mat_b.get_dimensions()[0] && w == mat_a.get_dimensions()[1] &&
         h == o_vec.get_dimensions()[0] && 1 == o_vec.get_dimensions()[1]);

  const int BLOCK_DIM = 256;
  const int GRID_DIM = calc_grid(h * WARP_SIZE, BLOCK_DIM);
  matrix_pair_mul_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat_a, h, w, pmat_b);
}

template <typename T>
__global__ void mm_1d(T* out_mat, const T* vec_a, int h, const T* vec_b, int w) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < h * w) {
    const int col = tid % w;
    const int row = tid / w;
    out_mat[tid] = vec_a[row] * vec_b[col];
  }
}

template <typename T>
void out_product(Tensor2<T>& out_mat, const Tensor2<T>& vec_a, const Tensor2<T>& vec_b,
                 cudaStream_t stream) {
  T* pout = out_mat.get_ptr();
  const T* pvec_a = vec_a.get_ptr();
  const T* pvec_b = vec_b.get_ptr();
  const auto& dim = out_mat.get_dimensions();

  const int h = dim[0];
  const int w = dim[1];

  assert(h == vec_a.get_dimensions()[0] && w == vec_b.get_dimensions()[1] &&
         vec_a.get_dimensions()[1] == 1 && vec_b.get_dimensions()[0] == 1);

  const int BLOCK_DIM = 256;
  const int GRID_DIM = calc_grid(h * w, BLOCK_DIM);
  mm_1d<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pvec_a, h, pvec_b, w);
}

/**
 * Each row in `mat` scale with the coresponding element in vec. and accum across rows
 * The length of vec should be h.
 * @param o_mat: hxw
 * @param mat: hxw
 * @param vec: hx1
 */
template <typename T>
__global__ void row_scaling_sum_kernel(T* out, const T* mat, int h, int w, const T* vec) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int wtid = tid % WARP_SIZE;  // thread id in warp
  const int wid = tid / WARP_SIZE;   // warp id
  if (wid < w) {
    T accum = 0.f;
    for (int i = wtid; i < h; i += WARP_SIZE) {
      const int col = wid;
      const int idx = i * w + col;
      accum += mat[idx] * vec[i];
    }
    T val = warpReduceSum(accum);
    if (wtid == 0) {
      out[wid] += val;  // using += here to enable regularization
    }
  }
}

template <typename T>
void row_scaling_sum(Tensor2<T>& out, const Tensor2<T>& mat, const Tensor2<T>& vec,
                     cudaStream_t stream) {
  T* pout = out.get_ptr();
  const T* pmat = mat.get_ptr();
  const T* pvec = vec.get_ptr();

  const auto& dim = out.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2 && idim[0] == vec.get_dimensions()[0] &&
         vec.get_dimensions()[1] == 1);
  assert(idim[1] == dim[1]);

  const int h = idim[0];
  const int w = idim[1];

  const int BLOCK_DIM = 256;
  const int GRID_DIM = calc_grid(w * WARP_SIZE, BLOCK_DIM);  // each col one warp

  row_scaling_sum_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w, pvec);
}

template <typename T>
void rows_sum(Tensor2<T>& out, const Tensor2<T>& mat, cudaStream_t stream) {
  T* pout = out.get_ptr();
  const T* pmat = mat.get_ptr();

  const auto& dim = out.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2);
  assert(idim[1] == dim[1]);

  const int h = idim[0];
  const int w = idim[1];

  MLCommon::LinAlg::reduce(pout, pmat, h, w, (T)0, false, true, stream, false,
                           [] __device__(T in, int i) { return in; });
}

}  // namespace

/*
 * Equivalent TensorFlow Code:
 *
def forward(x, k, b, layers):
  y = []
  h = []
  for i in range(layers):
    v = tf.linalg.matvec(x if i == 0 else y[i - 1], k[i])
    v = tf.transpose(v)
    h.append(v)
    m = tf.multiply(x, v)
    m = tf.add(m, x if i == 0 else y[i - 1])
    m = tf.add(m, b[i])
    y.append(m)
  return y, h
 *
 */
template <typename T>
void MultiCrossForwardFunctor<T>::operator()(
    cudaStream_t stream, cublasHandle_t cublas_handle, const Tensor2<T>& input_tensor,
    const Tensors2<T>& kernel_tensors, const Tensors2<T>& bias_tensors,
    Tensors2<T>& layer_output_tensors, Tensors2<T>& layer_hidden_tensors, int num_layers) const {
  for (int i = 0; i < num_layers; i++) {
    // weight: kernel_tensors[i] is a row vector
    // layer_hidden_tensors[i] is a row vector
    matrix_vec_mul(layer_hidden_tensors[i], i == 0 ? input_tensor : layer_output_tensors[i - 1],
                   kernel_tensors[i], cublas_handle, stream);
    row_scaling(layer_output_tensors[i], input_tensor, layer_hidden_tensors[i], stream);
    matrix_add(layer_output_tensors[i], layer_output_tensors[i],
               i == 0 ? input_tensor : layer_output_tensors[i - 1], stream);
    matrix_vec_add(layer_output_tensors[i], layer_output_tensors[i], bias_tensors[i], stream);
  }
}

//
/*
  ouput is x_{l+1} =  x_0 \. (w * x_l + b) + x_l , where
  input is
    input_tensor : x_0
    kernel_tensors : w
    bias_tensors   : n
    layer_output_tensors : x_l


  output is
    layer_output_tensors : x_l

  intermediate tensor:
    layer_hidden_tensors : w * x_l

h_i = gemv(x_i,w_i) ,
o_i = row_scaling(h_i,x),
o_i = matrix_vec_add(o_i,bias)
o_i = matrix_add(o_i,o_{i-1})

*
*/
template <typename T>

void MultiCrossForwardFunctorv2<T>::operator()(
    cudaStream_t stream, const Tensor2<T>& input_tensor, const Tensors2<T>& kernel_tensors,
    const Tensors2<T>& bias_tensors, Tensors2<T>& XU_tensors, Tensors2<T>& layer_output_tensors,
    Tensors2<T>& layer_hidden_tensors, int num_layers, const std::vector<CublasDesc<T>>& xu_descr_,
    const std::vector<CublasDesc<T>>& xuvb_descr_, const std::vector<CublasAlgo<T>>& xu_fprop_algo_,
    const std::vector<CublasAlgo<T>>& xuvb_fprop_algo_, cublasLtHandle_t cublaslt_handle) {
  auto batchsize = input_tensor.get_dimensions()[0];
  auto projection_dim = kernel_tensors[0].get_dimensions()[1];
  auto vec_length = input_tensor.get_dimensions()[1];
  auto U_row = kernel_tensors[0].get_dimensions()[0];
  auto V_col = kernel_tensors[1].get_dimensions()[1];
  float alpha = 1.0f;
  float beta = 0.0f;
  if (vec_length != U_row || vec_length != V_col) {
    HCTR_LOG(INFO, WORLD, "vec_length %d U_row %d V_col %d\n", vec_length, U_row, V_col);
    HCTR_OWN_THROW(Error_t::WrongInput, "input or output tensor dimensions not matches");
  }
  for (int i = 0; i < num_layers; i++) {
    const auto& tensor_input = i == 0 ? input_tensor : layer_output_tensors[i - 1];
    // gemm with functor
    // x_i * u
    {
      const T* mat_a = tensor_input.get_ptr();
      const T* mat_b = kernel_tensors[2 * i].get_ptr();
      T* mat_c = XU_tensors[i].get_ptr();
      this->gemm_functor_(alpha, mat_a, mat_b, beta, mat_c, mat_c, xu_descr_[i], xu_fprop_algo_[i],
                          cublaslt_handle, stream);
    }

    // gemm + bias with functor
    // x_i * u * v + b
    {
      const T* mat_a = XU_tensors[i].get_ptr();
      const T* mat_b = kernel_tensors[2 * i + 1].get_ptr();
      T* mat_c = layer_hidden_tensors[i].get_ptr();
      this->gemm_functor_(alpha, mat_a, mat_b, beta, mat_c, mat_c, xuvb_descr_[i],
                          xuvb_fprop_algo_[i], cublaslt_handle, stream);
    }
    // x_0 .* (x_i * u * v + b) + x_i
    fused_matrix_elementwise_dot_add(layer_output_tensors[i], layer_hidden_tensors[i], input_tensor,
                                     i == 0 ? input_tensor : layer_output_tensors[i - 1], stream);
  }
}

/*
 * Equivalent TensorFlow Code:
 *
def backward(x, k, y, h, dy, layers):
  dx = tf.zeros(x.shape)
  dk = []
  db = []
  for i in reversed(range(layers)):
    dx = tf.add(dx, tf.multiply(dy, h[i]))
    dv = tf.expand_dims(tf.reduce_sum(tf.multiply(dy, x), 1), 1)
    dk.insert(0, tf.linalg.matvec(x if i == 0 else y[i - 1], tf.transpose(dv), transpose_a=True))
    db.insert(0, tf.expand_dims(tf.reduce_sum(dy, 0), 0))
    dy = tf.add(dy, tf.matmul(dv, k[i]))
  dx = tf.add(dx, dy)
  return dx, dk, db
grad_tensor : dy
one multi-cross contains multiple cell:

tmp_mat_tensors[0] : dy * h[i]
tmp_mat_tensors[1] : tmp data gradient to current multicross cell
tmp_mat_tensors[2]: sum(dy/dh * h[i])
 *
 */
template <typename T>
void MultiCrossBackwardFunctor<T>::operator()(
    cudaStream_t stream, const Tensor2<T>& input_tensor, const Tensors2<T>& kernel_tensors,
    const Tensors2<T>& layer_output_tensors, const Tensors2<T>& layer_hidden_tensors,
    const Tensor2<T>& grad_tensor, Tensor2<T>& output_tensor, Tensors2<T>& kernel_output_tensors,
    Tensors2<T>& bias_output_tensors, Tensor2<T>& tmp_vec_tensor, Tensor2<T> tmp_mat_tensors[],
    int num_layers) const {
  cudaMemsetAsync(tmp_mat_tensors[2].get_ptr(), 0, tmp_mat_tensors[2].get_size_in_bytes(), stream);
  for (int i = num_layers - 1; i >= 0; i--) {
    // tmp_mat_tensors[0] = dy * h_i (h_i = gemv(x_i , w_i))
    row_scaling(tmp_mat_tensors[0], i == num_layers - 1 ? grad_tensor : tmp_mat_tensors[1],
                layer_hidden_tensors[i], stream);
    // dx
    matrix_add(tmp_mat_tensors[2], tmp_mat_tensors[2], tmp_mat_tensors[0], stream);
    // tmp_vec_tensor : {batchsize , 1}
    matrix_pair_mul(tmp_vec_tensor, i == num_layers - 1 ? grad_tensor : tmp_mat_tensors[1],
                    input_tensor, stream);

    // gemv(layer_output_tensors^T, tmp_vec_tensor)
    // gradient WRT weight
    row_scaling_sum(kernel_output_tensors[i], i == 0 ? input_tensor : layer_output_tensors[i - 1],
                    tmp_vec_tensor, stream);
    // dbias
    rows_sum(bias_output_tensors[i], i == num_layers - 1 ? grad_tensor : tmp_mat_tensors[1],
             stream);

    out_product(tmp_mat_tensors[0], tmp_vec_tensor, kernel_tensors[i], stream);
    matrix_add(tmp_mat_tensors[1], i == num_layers - 1 ? grad_tensor : tmp_mat_tensors[1],
               tmp_mat_tensors[0], stream);
  }
  matrix_add(output_tensor, tmp_mat_tensors[2], tmp_mat_tensors[1], stream);
}
template <typename T>
void MultiCrossBackwardFunctorv2<T>::operator()(
    cudaStream_t stream, const Tensor2<T>& input_tensor, const Tensors2<T>& kernel_tensors,
    const Tensors2<T>& layer_output_tensors, const Tensors2<T>& layer_hidden_tensors,
    const Tensor2<T>& grad_tensor, Tensor2<T>& output_tensor, Tensors2<T>& kernel_output_tensors,
    Tensors2<T>& bias_output_tensors, Tensors2<T>& XU_tensors, Tensor2<T> tmp_mat_tensors[],
    int num_layers, const std::vector<CublasDesc<T>>& xu_descr_,
    const std::vector<CublasDesc<T>>& xuvb_descr_,
    const std::vector<CublasDesc<T>>& du_descrs_bprop_,
    const std::vector<CublasDesc<T>>& dhidden_descrs_bprop_,
    const std::vector<CublasAlgo<T>>& xu_bprop_algo_,
    const std::vector<CublasAlgo<T>>& xuvb_bprop_algo_,
    const std::vector<CublasAlgo<T>>& du_bprop_algos_,
    const std::vector<CublasAlgo<T>>& dhidden_bprop_algos_, cublasLtHandle_t cublaslt_handle) {
  cudaMemsetAsync(tmp_mat_tensors[2].get_ptr(), 0, tmp_mat_tensors[2].get_size_in_bytes(), stream);
  auto batchsize = input_tensor.get_dimensions()[0];
  auto projection_dim = kernel_tensors[0].get_dimensions()[1];
  auto vec_length = input_tensor.get_dimensions()[1];
  auto U_row = kernel_tensors[0].get_dimensions()[0];
  auto V_col = kernel_tensors[1].get_dimensions()[1];
  for (int i = num_layers - 1; i >= 0; i--) {
    // S0 = dY_i .* X , shape: (batchsize, w)
    // dX += dY_i .* H , shape: (batchsize, w)
    fused_mul_fma3(tmp_mat_tensors[0], tmp_mat_tensors[2],
                   i == num_layers - 1 ? grad_tensor : tmp_mat_tensors[1], input_tensor,
                   layer_hidden_tensors[i], stream);
    {
      // 1 db, dV = XU_{i}^T * S0 shape: (project_dim, w)
      const T* mat_a = XU_tensors[i].get_ptr();
      const T* mat_b = tmp_mat_tensors[0].get_ptr();
      T* mat_c = kernel_output_tensors[2 * i + 1].get_ptr();
      this->gemm_functor_(1.0f, mat_a, mat_b, 1.0f, mat_c, mat_c, xu_descr_[i], xu_bprop_algo_[i],
                          cublaslt_handle, stream);
      // 2 dH = S1 = S0 * V^T shape: (batchsize, project_dim)
      mat_a = tmp_mat_tensors[0].get_ptr();
      mat_b = kernel_tensors[2 * i + 1].get_ptr();
      mat_c = tmp_mat_tensors[3].get_ptr();
      this->gemm_functor_(1.0f, mat_a, mat_b, 0.0f, mat_c, mat_c, xuvb_descr_[i],
                          xuvb_bprop_algo_[i], cublaslt_handle, stream);
      // 3  dU = X_{i-1} ^T * S1 shape: (w, project_dim)
      mat_a = i == 0 ? input_tensor.get_ptr() : layer_output_tensors[i - 1].get_ptr();
      mat_b = tmp_mat_tensors[3].get_ptr();
      mat_c = kernel_output_tensors[2 * i].get_ptr();
      this->gemm_functor_(1.0f, mat_a, mat_b, 1.0f, mat_c, mat_c, du_descrs_bprop_[i],
                          du_bprop_algos_[i], cublaslt_handle, stream);
      // 4 dY_{i-1} = S1 * U^T + dY_{i} shape: (batchsize, w)
      mat_a = tmp_mat_tensors[3].get_ptr();
      mat_b = kernel_tensors[i * 2].get_ptr();
      auto dgrad = (i == num_layers - 1 ? grad_tensor : tmp_mat_tensors[1]);
      mat_c = dgrad.get_ptr();
      T* mat_d = tmp_mat_tensors[1].get_ptr();
      // gemm: mat_d = mat_a * mat_b + mat_c
      this->gemm_functor_(1.0f, mat_a, mat_b, 1.0f, mat_c, mat_d, dhidden_descrs_bprop_[i],
                          dhidden_bprop_algos_[i], cublaslt_handle, stream);
    }
  }
  matrix_add(output_tensor, tmp_mat_tensors[2], tmp_mat_tensors[1], stream);
}

template <typename T>
MultiCrossLayer<T>::MultiCrossLayer(
    const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
    const std::shared_ptr<BufferBlock2<T>>& weight_buff,
    const std::shared_ptr<BufferBlock2<T>>& wgrad_buff,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff, const Tensor2<T>& in_tensor,
    const Tensor2<T>& out_tensor, const std::shared_ptr<GPUResource>& gpu_resource, int num_layers,
    size_t projection_dim, std::vector<Initializer_t> initializer_types, bool enable_tf32_compute)
    : TrainableLayer<T>(master_weight_buff, weight_buff, wgrad_buff, gpu_resource,
                        initializer_types),
      num_layers_(num_layers),
      projection_dim_(projection_dim),
      enable_tf32_compute_(enable_tf32_compute) {
  try {
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor.get_dimensions();
    const auto& out_tensor_dim = out_tensor.get_dimensions();
    size_t vec_length = in_tensor_dim[1];
    size_t batchsize = in_tensor_dim[0];
    if (projection_dim_ == 0) {
      HCTR_LOG(WARNING, ROOT, "using multi-cross v1\n");
    }
    // 1. two dim?
    if (in_tensor_dim.size() != 2 || out_tensor_dim.size() != 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
    }
    // 2. same dim?
    for (int i = 0; i < 2; i++) {
      if (in_tensor_dim[i] != out_tensor_dim[i]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "input and output tensor doesn't match");
      }
    }

    // check num_lyaers
    if (num_layers < 1) {
      HCTR_OWN_THROW(Error_t::WrongInput, "num_layers < 1");
    }

    std::vector<size_t> bias_dim = {1, vec_length};
    std::vector<size_t> weight_dim = {vec_length, vec_length};
    std::vector<size_t> U_dim = {vec_length, this->projection_dim_};
    std::vector<size_t> V_dim = {this->projection_dim_, vec_length};
    if (!this->projection_dim_) {
      weight_dim[0] = 1ul;
    }
    for (int i = 0; i < num_layers; i++) {
      // setup weights and bias
      {
        // dcnv2
        if (this->projection_dim_) {
          //  Tensor2<T> U, V;
          //  weight_buff->reserve(U_dim, &U);
          //  weight_buff->reserve(V_dim, &V);
          //  weights_.push_back(U);
          //  weights_.push_back(V);
          this->set_weight(3 * i, U_dim);
          this->set_weight(3 * i + 1, V_dim);
          this->set_weight(3 * i + 2, bias_dim);
          // dcnv1
        } else {
          //  Tensor2<T> tensor;
          //  weight_buff->reserve(weight_dim, &tensor);
          //  weights_.push_back(tensor);
          this->set_weight(2 * i, weight_dim);
          this->set_weight(2 * i + 1, bias_dim);
        }
      }
      // setup weight gradient
      // dcnv2
      if (this->projection_dim_) {
        //  Tensor2<T> U, V;
        //  wgrad_buff->reserve(U_dim, &U);
        //  wgrad_buff->reserve(V_dim, &V);
        //  wgrad_.push_back(U);
        //  wgrad_.push_back(V);
        this->set_wgrad(3 * i, U_dim);
        this->set_wgrad(3 * i + 1, V_dim);
        this->set_wgrad(3 * i + 2, bias_dim);
        // dcnv1
      } else {
        this->set_wgrad(2 * i, weight_dim);
        this->set_wgrad(2 * i + 1, bias_dim);
      }

      if (this->projection_dim_) {
        xu_descrs_fprop_.emplace_back();
        xuvb_descrs_fprop_.emplace_back();
        xu_descrs_bprop_.emplace_back();
        xuvb_descrs_bprop_.emplace_back();
        du_descrs_bprop_.emplace_back();
        dhidden_descrs_bprop_.emplace_back();

        xu_fprop_algos_.emplace_back();
        xuvb_fprop_algos_.emplace_back();
        xu_bprop_algos_.emplace_back();
        xuvb_bprop_algos_.emplace_back();
        du_bprop_algos_.emplace_back();
        dhidden_bprop_algos_.emplace_back();
      }
    }

    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);
    // setup blobs

    std::vector<size_t> blob_dim = {batchsize, vec_length};

    // input
    blob_tensors_.push_back(in_tensor);
    // intermediate output
    for (int i = 0; i < num_layers - 1; i++) {
      Tensor2<T> tensor;
      blobs_buff->reserve(blob_dim, &tensor);
      blob_tensors_.push_back(tensor);
    }
    // output
    blob_tensors_.push_back(out_tensor);

    for (int i = 0; i < 3; i++) {
      blobs_buff->reserve(blob_dim, &tmp_mat_tensors_[i]);
    }
    if (projection_dim_) {
      blobs_buff->reserve({batchsize, projection_dim_}, &tmp_mat_tensors_[3]);
    }
    std::vector<size_t> tmp_vec_dim = {batchsize, 1};
    std::vector<size_t> hidden_dim = {batchsize, weight_dim[0]};
    blobs_buff->reserve(tmp_vec_dim, &tmp_vec_tensor_);
    if (this->projection_dim_) {
      std::vector<size_t> XU_dim = {batchsize, this->projection_dim_};
      for (int i = 0; i < num_layers; i++) {
        Tensor2<T> tensor;
        blobs_buff->reserve(XU_dim, &tensor);
        XU_tensors_.push_back(tensor);
      }
    }
    for (int i = 0; i < num_layers; i++) {
      Tensor2<T> tensor;
      blobs_buff->reserve(hidden_dim, &tensor);
      hidden_tensors_.push_back(tensor);
    }
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void MultiCrossLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(this->get_device_id());
  Tensors2<T> kernel_tensors;
  Tensors2<T> bias_tensors;
  Tensors2<T> output_tensors;
  Tensors2<T> hidden_tensors;

  if (this->projection_dim_) {
    for (int i = 0; i < num_layers_; i++) {
      kernel_tensors.push_back(this->get_weight(3 * i));
      kernel_tensors.push_back(this->get_weight(3 * i + 1));
      bias_tensors.push_back(this->get_weight(3 * i + 2));
    }
  } else {
    for (int i = 0; i < num_layers_; i++) {
      kernel_tensors.push_back(this->get_weight(2 * i));
      bias_tensors.push_back(this->get_weight(2 * i + 1));
    }
  }
  for (int i = 0; i < num_layers_; i++) {
    output_tensors.push_back(blob_tensors_[i + 1]);
    hidden_tensors.push_back(hidden_tensors_[i]);
  }
  if (this->projection_dim_ == 0) {
    // dcn v1
    MultiCrossForwardFunctor<T>()(this->get_gpu().get_stream(), this->get_gpu().get_cublas_handle(),
                                  blob_tensors_[0], kernel_tensors, bias_tensors, output_tensors,
                                  hidden_tensors, num_layers_);
  } else {
    // dcn v2
    this->dcnv2_forward_functor_(this->get_gpu().get_stream(), blob_tensors_[0], kernel_tensors,
                                 bias_tensors, XU_tensors_, output_tensors, hidden_tensors,
                                 num_layers_, xu_descrs_fprop_, xuvb_descrs_fprop_, xu_fprop_algos_,
                                 xuvb_fprop_algos_, this->get_gpu().get_cublaslt_handle());
  }
}

template <typename T>
void MultiCrossLayer<T>::bprop() {
  CudaDeviceContext context(this->get_device_id());
  Tensors2<T> kernel_tensors;
  Tensors2<T> kernel_output_tensors;
  Tensors2<T> bias_output_tensors;
  Tensors2<T> forward_output_tensors;
  Tensors2<T> forward_hidden_tensors;
  // dcnv2
  if (this->projection_dim_) {
    for (int i = 0; i < num_layers_; i++) {
      // U
      kernel_tensors.push_back(this->get_weight(3 * i));
      // V
      kernel_tensors.push_back(this->get_weight(3 * i + 1));
      // dU
      kernel_output_tensors.push_back(this->get_wgrad(3 * i));
      // dV
      kernel_output_tensors.push_back(this->get_wgrad(3 * i + 1));
      // db
      bias_output_tensors.push_back(this->get_wgrad(3 * i + 2));
      // intermediate output
      forward_hidden_tensors.push_back(hidden_tensors_[i]);
    }
  } else {
    for (int i = 0; i < num_layers_; i++) {
      kernel_tensors.push_back(this->get_weight(2 * i));
      kernel_output_tensors.push_back(this->get_wgrad(2 * i));
      bias_output_tensors.push_back(this->get_wgrad(2 * i + 1));
      forward_hidden_tensors.push_back(hidden_tensors_[i]);
    }
  }

  for (int i = 0; i < num_layers_ - 1; i++) {
    forward_output_tensors.push_back(blob_tensors_[i + 1]);
  }
  if (this->projection_dim_ == 0) {
    // dcn v1
    MultiCrossBackwardFunctor<T>()(
        this->get_gpu().get_stream(), blob_tensors_[0], kernel_tensors, forward_output_tensors,
        forward_hidden_tensors, blob_tensors_[num_layers_], blob_tensors_[0], kernel_output_tensors,
        bias_output_tensors, tmp_vec_tensor_, tmp_mat_tensors_, num_layers_);
  } else {
    // dcn v2
    this->dcnv2_backward_functor_(
        this->get_gpu().get_stream(), blob_tensors_[0], kernel_tensors, forward_output_tensors,
        forward_hidden_tensors, blob_tensors_[num_layers_], blob_tensors_[0], kernel_output_tensors,
        bias_output_tensors, this->XU_tensors_, tmp_mat_tensors_, num_layers_, xu_descrs_bprop_,
        xuvb_descrs_bprop_, du_descrs_bprop_, dhidden_descrs_bprop_, xu_bprop_algos_,
        xuvb_bprop_algos_, du_bprop_algos_, dhidden_bprop_algos_,
        this->get_gpu().get_cublaslt_handle());
  }
}
template <typename T>
void MultiCrossLayer<T>::search_algorithm() {
  // dcnv1 no search_algorithm
  CudaDeviceContext context(this->get_device_id());
  auto cublaslt_handle = this->get_gpu().get_cublaslt_handle();
  auto stream = this->get_gpu().get_stream();
  if (this->projection_dim_) {
    // setting up for fprop()
    {
      for (int i = 0; i < num_layers_; i++) {
        const auto& tensor_input = blob_tensors_[i];
        const T* mat_a = tensor_input.get_ptr();
        const T* mat_b = this->get_weight(3 * i).get_ptr();
        T* mat_c = XU_tensors_[i].get_ptr();

        this->xu_fprop_algos_[i].search_algorithm(1.0f, mat_a, mat_b, 0.f, mat_c, mat_c,
                                                  xu_descrs_fprop_[i], cublaslt_handle, stream);
        mat_a = XU_tensors_[i].get_ptr();
        mat_b = this->get_weight(3 * i + 1).get_ptr();
        mat_c = hidden_tensors_[i].get_ptr();
        this->xuvb_fprop_algos_[i].search_algorithm(1.0f, mat_a, mat_b, 0.f, mat_c, mat_c,
                                                    xuvb_descrs_fprop_[i], cublaslt_handle, stream);
      }
    }

    // setting up for bprop()
    {
      for (int i = 0; i < num_layers_; i++) {
        // 1
        const T* mat_a = XU_tensors_[i].get_ptr();
        const T* mat_b = tmp_mat_tensors_[0].get_ptr();
        T* mat_c = this->get_wgrad(3 * i + 1).get_ptr();
        this->xu_bprop_algos_[i].search_algorithm(1.0, mat_a, mat_b, 1.0, mat_c, mat_c,
                                                  xu_descrs_bprop_[i], cublaslt_handle, stream);
        // 2
        mat_a = tmp_mat_tensors_[0].get_ptr();
        mat_b = this->get_wgrad(3 * i + 1).get_ptr();
        mat_c = tmp_mat_tensors_[3].get_ptr();
        this->xuvb_bprop_algos_[i].search_algorithm(1.0, mat_a, mat_b, 0.0, mat_c, mat_c,
                                                    xuvb_descrs_bprop_[i], cublaslt_handle, stream);
        // 3
        mat_a = blob_tensors_[i].get_ptr();
        mat_b = tmp_mat_tensors_[3].get_ptr();
        mat_c = this->get_wgrad(3 * i).get_ptr();
        this->du_bprop_algos_[i].search_algorithm(1.0, mat_a, mat_b, 1.0, mat_c, mat_c,
                                                  du_descrs_bprop_[i], cublaslt_handle, stream);

        // 4
        mat_a = tmp_mat_tensors_[3].get_ptr();
        mat_b = this->get_weight(3 * i).get_ptr();
        mat_c = tmp_mat_tensors_[0].get_ptr();
        this->dhidden_bprop_algos_[i].search_algorithm(1.0, mat_a, mat_b, 1.0, mat_c, mat_c,
                                                       dhidden_descrs_bprop_[i], cublaslt_handle,
                                                       stream);
      }
    }
  }
}
template <typename T>
void MultiCrossLayer<T>::initialize() {
  auto cublaslt_handle = this->get_gpu().get_cublaslt_handle();
  auto stream = this->get_gpu().get_stream();
  if (this->projection_dim_) {
    // setting up for fprop()
    {
      for (int i = 0; i < num_layers_; i++) {
        const auto& tensor_input = blob_tensors_[i];
        std::vector<size_t> dims_a(tensor_input.get_dimensions());
        std::vector<size_t> dims_b(this->get_weight(3 * i).get_dimensions());
        this->xu_descrs_fprop_[i].set_fprop_attr(dims_a, dims_b, CUBLAS_OP_N, CUBLAS_OP_N,
                                                 CUBLASLT_ORDER_ROW, this->enable_tf32_compute_,
                                                 nullptr);
        this->xu_fprop_algos_[i].init_algorithm(this->xu_descrs_fprop_[i], cublaslt_handle);

        dims_a = XU_tensors_[i].get_dimensions();
        dims_b = this->get_weight(3 * i + 1).get_dimensions();
        T* bias = this->get_weight(3 * i + 2).get_ptr();

        this->xuvb_descrs_fprop_[i].set_fprop_attr(dims_a, dims_b, CUBLAS_OP_N, CUBLAS_OP_N,
                                                   CUBLASLT_ORDER_ROW, this->enable_tf32_compute_,
                                                   bias);
        this->xuvb_fprop_algos_[i].init_algorithm(this->xuvb_descrs_fprop_[i], cublaslt_handle);
      }
    }
    // setting up for bprop()
    {
      for (int i = 0; i < num_layers_; i++) {
        // 1
        std::vector<size_t> dims_a(XU_tensors_[i].get_dimensions());
        std::vector<size_t> dims_b(tmp_mat_tensors_[0].get_dimensions());
        T* dbias = this->get_wgrad(3 * i + 2).get_ptr();
        this->xu_descrs_bprop_[i].set_bprop_attr(dims_a, dims_b, CUBLAS_OP_T, CUBLAS_OP_N,
                                                 CUBLASLT_ORDER_ROW, this->enable_tf32_compute_,
                                                 dbias);
        this->xu_bprop_algos_[i].init_algorithm(this->xu_descrs_bprop_[i], cublaslt_handle);

        // 2
        dims_a = tmp_mat_tensors_[0].get_dimensions();
        dims_b = this->get_weight(3 * i + 1).get_dimensions();
        this->xuvb_descrs_bprop_[i].set_bprop_attr(dims_a, dims_b, CUBLAS_OP_N, CUBLAS_OP_T,
                                                   CUBLASLT_ORDER_ROW, this->enable_tf32_compute_);
        this->xuvb_bprop_algos_[i].init_algorithm(this->xuvb_descrs_bprop_[i], cublaslt_handle);

        // 3
        dims_a = blob_tensors_[i].get_dimensions();
        dims_b = XU_tensors_[i].get_dimensions();
        this->du_descrs_bprop_[i].set_bprop_attr(dims_a, dims_b, CUBLAS_OP_T, CUBLAS_OP_N,
                                                 CUBLASLT_ORDER_ROW, this->enable_tf32_compute_);
        this->du_bprop_algos_[i].init_algorithm(this->du_descrs_bprop_[i], cublaslt_handle);

        // 4
        dims_a = XU_tensors_[i].get_dimensions();
        dims_b = this->get_weight(3 * i).get_dimensions();
        this->dhidden_descrs_bprop_[i].set_bprop_attr(dims_a, dims_b, CUBLAS_OP_N, CUBLAS_OP_T,
                                                      CUBLASLT_ORDER_ROW,
                                                      this->enable_tf32_compute_);
        this->dhidden_bprop_algos_[i].init_algorithm(this->dhidden_descrs_bprop_[i],
                                                     cublaslt_handle);
      }
    }
  }
}
template <typename T>
std::unique_ptr<DataSimulator> MultiCrossLayer<T>::get_default_initializer(const int index) {
  const Tensor2<T>& in_tensor = in_tensors_[0];
  const Tensor2<T>& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor.get_dimensions()[1];
  float top_dim = out_tensor.get_dimensions()[1];

  std::unique_ptr<DataSimulator> simu(nullptr);
  int idx = -1;
  // each dcn2 layer has one more weight tensor (U and V)
  // U V shares the same initializer
  if (this->projection_dim_) {
    idx = index % 3 == 2 ? 1 : 0;
  } else {
    idx = index % 2;
  }
  // weight
  if (0 == idx) {
    // aligned with pytorch: xavier_norm
    simu.reset(new VarianceScalingSimulator(1.f, data_simu::Mode_t::Fan_avg,
                                            data_simu::Distribution_t::Norm, bottom_dim, top_dim));
  } else if (1 == idx) {
    // aligned with pytorch: zero
    simu.reset(new ConstantDataSimulator(0.0f));
  } else {
    HCTR_OWN_THROW(Error_t::OutOfBound, "index != {0, 1, 2}.");
  }
  return simu;
}

template class MultiCrossLayer<float>;
template class MultiCrossLayer<__half>;

}  // namespace HugeCTR

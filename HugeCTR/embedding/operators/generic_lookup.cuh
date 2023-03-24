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

#include <embedding/common.hpp>
#include <embedding/view.hpp>
#include <utils.cuh>

namespace embedding {

DEVICE_INLINE void float_half_atomicAdd_lower_cuda(__half *dst, float value) {
  bool uplo = ((unsigned long long)dst) & 2;  // check if the atomic is for the upper or lower
                                              // 16-bit quantity in the aligned 32-bit item
  unsigned *addr = reinterpret_cast<unsigned *>(
      ((unsigned long long)dst) & 0xFFFFFFFFFFFFFFFCULL);  // get the 32-bit aligned address
  unsigned old = *addr;
  unsigned val;
  do {
    val = old;
    float newval = __half2float(__ushort_as_half(uplo ? ((unsigned short)(val >> 16))
                                                      : ((unsigned short)(val)))) +
                   value;
    unsigned short newval_s = __half_as_ushort(__float2half(newval));
    unsigned newval_u = val & (uplo ? (0x0FFFFU) : (0xFFFF0000U));
    newval_u |= uplo ? (((unsigned)newval_s) << 16) : (newval_s);
    old = atomicCAS(addr, old, newval_u);
  } while (old != val);
  return;
}

template <typename T>
struct Vec4T {};

template <>
struct Vec4T<__half> {
  union U {
    float2 f;
    __half2 h[2];
  } value;

  DEVICE_INLINE Vec4T() {
    value.h[0].x = 0.f;
    value.h[0].y = 0.f;
    value.h[1].x = 0.f;
    value.h[1].y = 0.f;
  }

  DEVICE_INLINE void reset() {
    value.h[0].x = 0.f;
    value.h[0].y = 0.f;
    value.h[1].x = 0.f;
    value.h[1].y = 0.f;
  }

  DEVICE_INLINE void load(const float *p, int n) {
    if (n == 4) {
      float4 f = *(reinterpret_cast<const float4 *>(p));
      float2 firstf{f.x, f.y};
      float2 secondf{f.z, f.w};
      value.h[0] = __float22half2_rn(firstf);
      value.h[1] = __float22half2_rn(secondf);
    } else {
      if (n > 0) value.h[0].x = __float2half(p[0]);
      if (n > 1) value.h[0].y = __float2half(p[1]);
      if (n > 2) value.h[1].x = __float2half(p[2]);
    }
    // if (n > 0)value.h[0].x = __float2half(p[0]);
    // if (n > 1)  value.h[0].y = __float2half(p[1]);
    // if (n > 2)  value.h[1].x = __float2half(p[2]);
    // if (n > 3)  value.h[1].y = __float2half(p[3]);
  }

  DEVICE_INLINE void load(const __half *p, int n) {
    if (n == 4) {
      value.f = *(reinterpret_cast<const float2 *>(p));
    } else {
      if (n > 0) value.h[0].x = p[0];
      if (n > 1) value.h[0].y = p[1];
      if (n > 2) value.h[1].x = p[2];
    }
  }

  DEVICE_INLINE void store(float *dst, int n) {
    if (n == 4) {
      float4 f;
      f.x = __half2float(value.h[0].x);
      f.y = __half2float(value.h[0].y);
      f.z = __half2float(value.h[1].x);
      f.w = __half2float(value.h[1].y);
      *(reinterpret_cast<float4 *>(dst)) = f;
    } else {
      if (n > 0) dst[0] = __half2float(value.h[0].x);
      if (n > 1) dst[1] = __half2float(value.h[0].y);
      if (n > 2) dst[2] = __half2float(value.h[1].x);
    }
  }

  DEVICE_INLINE void store(__half *dst, int n) {
    if (n == 4) {
      *(reinterpret_cast<float2 *>(dst)) = value.f;
    } else {
      if (n > 0) dst[0] = value.h[0].x;
      if (n > 1) dst[1] = value.h[0].y;
      if (n > 2) dst[2] = value.h[1].x;
    }
  }

  DEVICE_INLINE void atomic_store_accum(float *dst, int n) {
    if (n > 0) atomicAdd(dst, __half2float(value.h[0].x));
    if (n > 1) atomicAdd(dst + 1, __half2float(value.h[0].y));
    if (n > 2) atomicAdd(dst + 2, __half2float(value.h[1].x));
    if (n > 3) atomicAdd(dst + 3, __half2float(value.h[1].y));
  }

  DEVICE_INLINE void atomic_store_accum(__half *dst, int n) {
#if __CUDA_ARCH__ >= 700
    if (n == 4) {
      atomicAdd((reinterpret_cast<__half2 *>(dst)), value.h[0]);
      atomicAdd((reinterpret_cast<__half2 *>(dst + 2)), value.h[1]);
    } else {
      if (n > 0) atomicAdd(dst, value.h[0].x);
      if (n > 1) atomicAdd(dst + 1, value.h[0].y);
      if (n > 2) atomicAdd(dst + 2, value.h[1].x);
      if (n > 3) atomicAdd(dst + 3, value.h[1].y);
    }
#else
    if (n > 0) float_half_atomicAdd_lower_cuda(dst, __half2float(value.h[0].x));
    if (n > 1) float_half_atomicAdd_lower_cuda(dst + 1, __half2float(value.h[0].y));
    if (n > 2) float_half_atomicAdd_lower_cuda(dst + 2, __half2float(value.h[1].x));
    if (n > 3) float_half_atomicAdd_lower_cuda(dst + 3, __half2float(value.h[1].y));
#endif
  }
};

template <>
struct Vec4T<float> {
  float4 val;

  DEVICE_INLINE Vec4T() {
    val.x = 0.f;
    val.y = 0.f;
    val.z = 0.f;
    val.w = 0.f;
  }

  DEVICE_INLINE void reset() {
    val.x = 0.f;
    val.y = 0.f;
    val.z = 0.f;
    val.w = 0.f;
  }

  DEVICE_INLINE void load(const float *p, int n) {
    if (n == 4) {
      val = *((const float4 *)p);
    } else {
      if (n > 0) val.x = p[0];
      if (n > 1) val.y = p[1];
      if (n > 2) val.z = p[2];
    }
  }

  DEVICE_INLINE void load(const __half *p, int n) {
    if (n == 4) {
      Vec4T<__half> h;
      h.load(p, n);
      val.x = __half2float(h.value.h[0].x);
      val.y = __half2float(h.value.h[0].y);
      val.z = __half2float(h.value.h[1].x);
      val.w = __half2float(h.value.h[1].y);
    } else {
      if (n > 0) val.x = __half2float(p[0]);
      if (n > 1) val.y = __half2float(p[1]);
      if (n > 2) val.z = __half2float(p[2]);
    }
  }

  DEVICE_INLINE void store(float *dst, int n) {
    if (n == 4) {
      *(reinterpret_cast<float4 *>(dst)) = val;
    } else {
      if (n > 0) dst[0] = val.x;
      if (n > 1) dst[1] = val.y;
      if (n > 2) dst[2] = val.z;
    }
  }

  DEVICE_INLINE void store(__half *dst, int n) {
    if (n == 4) {
      Vec4T<__half> h;
      h.load(reinterpret_cast<float *>(&val), 4);
      h.store(dst, 4);
    } else {
      if (n > 0) dst[0] = __float2half(val.x);
      if (n > 1) dst[1] = __float2half(val.y);
      if (n > 2) dst[2] = __float2half(val.z);
    }
  }

  DEVICE_INLINE void atomic_store_accum(float *dst, int n) {
    if (n > 0) atomicAdd(dst, val.x);
    if (n > 1) atomicAdd(dst + 1, val.y);
    if (n > 2) atomicAdd(dst + 2, val.z);
    if (n > 3) atomicAdd(dst + 3, val.w);
  }

  DEVICE_INLINE void atomic_store_accum(__half *dst, int n) {
#if __CUDA_ARCH__ >= 700
    if (n == 4) {
      __half2 tmp1;
      __half2 tmp2;
      tmp1.x = __float2half(val.x);
      tmp1.y = __float2half(val.y);
      tmp2.x = __float2half(val.z);
      tmp2.y = __float2half(val.w);
      atomicAdd((reinterpret_cast<__half2 *>(dst)), tmp1);
      atomicAdd((reinterpret_cast<__half2 *>(dst + 2)), tmp2);
    } else {
      if (n > 0) atomicAdd(dst, __float2half(val.x));
      if (n > 1) atomicAdd(dst + 1, __float2half(val.y));
      if (n > 2) atomicAdd(dst + 2, __float2half(val.z));
      if (n > 3) atomicAdd(dst + 3, __float2half(val.w));
    }
#else
    if (n > 0) float_half_atomicAdd_lower_cuda(dst, val.x);
    if (n > 1) float_half_atomicAdd_lower_cuda(dst + 1, val.y);
    if (n > 2) float_half_atomicAdd_lower_cuda(dst + 2, val.z);
    if (n > 3) float_half_atomicAdd_lower_cuda(dst + 3, val.w);
#endif
  }

  DEVICE_INLINE void accumulate(const Vec4T<float> &other) {
    val.x += other.val.x;
    val.y += other.val.y;
    val.z += other.val.z;
    val.w += other.val.w;
  }

  DEVICE_INLINE void accumulate(const Vec4T<__half> &other) {
    val.x += __half2float(other.value.h[0].x);
    val.y += __half2float(other.value.h[0].y);
    val.z += __half2float(other.value.h[1].x);
    val.w += __half2float(other.value.h[1].y);
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<float> &other, float weight) {
    val.x += (other.val.x * weight);
    val.y += (other.val.y * weight);
    val.z += (other.val.z * weight);
    val.w += (other.val.w * weight);
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<float> &other, __half weight) {
    val.x += (other.val.x * __half2float(weight));
    val.y += (other.val.y * __half2float(weight));
    val.z += (other.val.z * __half2float(weight));
    val.w += (other.val.w * __half2float(weight));
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<__half> &other, float weight) {
    val.x += (__half2float(other.value.h[0].x) * weight);
    val.y += (__half2float(other.value.h[0].y) * weight);
    val.z += (__half2float(other.value.h[1].x) * weight);
    val.w += (__half2float(other.value.h[1].y) * weight);
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<__half> &other, __half weight) {
    val.x += (__half2float(other.value.h[0].x) * __half2float(weight));
    val.y += (__half2float(other.value.h[0].y) * __half2float(weight));
    val.z += (__half2float(other.value.h[1].x) * __half2float(weight));
    val.w += (__half2float(other.value.h[1].y) * __half2float(weight));
  }
};

#define NUM_VECTOR_PER_WARP 32
inline void get_kernel_config_use_warp(const int num_sms, const int num_thread_per_sm,
                                       const int block_size, const int warp_size,
                                       const int num_vector, int *grid_size,
                                       int *num_vector_per_warp, const int multiple_num = 4) {
  int warp_num_per_sm = num_thread_per_sm / warp_size;
  int warp_num_per_block = block_size / warp_size;
  int saturate_num = num_sms * warp_num_per_sm * multiple_num;

  if (num_vector <= saturate_num) {
    *num_vector_per_warp = 1;
    *grid_size = (num_vector - 1) / warp_num_per_block + 1;
    return;
  }

  if (num_vector / saturate_num >= NUM_VECTOR_PER_WARP) {
    *num_vector_per_warp = NUM_VECTOR_PER_WARP;
    *grid_size = (num_vector - 1) / (NUM_VECTOR_PER_WARP * warp_num_per_block) + 1;
  } else {
    *num_vector_per_warp = num_vector / saturate_num + 1;
    *grid_size = (saturate_num - 1) / warp_num_per_block + 1;
  }
  return;
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void multi_to_one_cta_per_ev_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;
  int i_ev = blockIdx.x;

  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    int average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);
    dst_type *dst_ev = copy_desc.get_dst_ptr(i_ev);

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);

    float accum[kMaxElemPerThread] = {0.f};
    for (int r = 0; r < (end - start); ++r) {
      const src_type *src_ev = copy_desc.get_src_ptr(r + start);
#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < vec_length; ++i) {
        accum[i] += HugeCTR::TypeConvertFunc<float, src_type>::convert(
            src_ev[blockDim.x * i + threadIdx.x]);
      }
    }
#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i] /= average_pooling_factor;
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < vec_length; ++i) {
      dst_ev[blockDim.x * i + threadIdx.x] =
          HugeCTR::TypeConvertFunc<dst_type, float>::convert(accum[i]);
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void multi_to_one_warp_per_ev_vec4_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int i_ev = blockIdx.x * blockDim.y + warp_id;
  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    int average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);

    dst_type *dst_ev = copy_desc.get_dst_ptr(i_ev);

    Vec4T<float> accum[kMaxElemPerThread];
    int L = end - start;
    for (int r = 0; r < L; r += kWarpSize) {
      int l = r + lane_id < L ? start + r + lane_id : 0;

      for (int j = 0; j < kWarpSize && r + j < L; ++j) {
        int j_ev = __shfl_sync(0xFFFFFFFF, l, j);
        const src_type *src_ev = copy_desc.get_src_ptr(j_ev);

#pragma unroll kMaxElemPerThread
        for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length;
             ++i) {
          Vec4T<src_type> src_elem;
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(vec_length - idx4, copy_width);
          src_elem.load(src_ev + idx4, n);
          accum[i].accumulate(src_elem);
        }
      }
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i].val.x /= average_pooling_factor;
      accum[i].val.y /= average_pooling_factor;
      accum[i].val.z /= average_pooling_factor;
      accum[i].val.w /= average_pooling_factor;
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].store(dst_ev + idx4, n);
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void multi_to_one_weight_cta_per_ev_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;
  int i_ev = blockIdx.x;

  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    float average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);
    dst_type *dst_ev = copy_desc.get_dst_ptr(i_ev);

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);

    float accum[kMaxElemPerThread] = {0.f};
    for (int r = 0; r < (end - start); ++r) {
      const src_type *src_ev = copy_desc.get_src_ptr(r + start);
      const float weight = copy_desc.get_sp_weight(r + start);
#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < vec_length; ++i) {
        accum[i] += HugeCTR::TypeConvertFunc<float, src_type>::convert(
                        src_ev[blockDim.x * i + threadIdx.x]) /
                    weight;
      }
    }
#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i] /= average_pooling_factor;
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < vec_length; ++i) {
      dst_ev[blockDim.x * i + threadIdx.x] =
          HugeCTR::TypeConvertFunc<dst_type, float>::convert(accum[i]);
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void multi_to_one_warp_per_ev_vec4_less_block_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  for (int i_ev = blockIdx.x * blockDim.y + warp_id; i_ev < copy_desc.num_vec_;
       i_ev += gridDim.x * blockDim.y) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    int average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);

    dst_type *dst_ev = copy_desc.get_dst_ptr(i_ev);

    Vec4T<float> accum[kMaxElemPerThread];
    int L = end - start;
    for (int r = 0; r < L; ++r) {
      const src_type *src_ev = copy_desc.get_src_ptr(start + r);
#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
        Vec4T<src_type> src_elem;
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min(vec_length - idx4, copy_width);
        src_elem.load(src_ev + idx4, n);
        accum[i].accumulate(src_elem);
      }
    }
#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i].val.x /= average_pooling_factor;
      accum[i].val.y /= average_pooling_factor;
      accum[i].val.z /= average_pooling_factor;
      accum[i].val.w /= average_pooling_factor;
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].store(dst_ev + idx4, n);
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void multi_to_one_weight_warp_per_ev_vec4_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int i_ev = blockIdx.x * blockDim.y + warp_id;
  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    float average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);

    dst_type *dst_ev = copy_desc.get_dst_ptr(i_ev);

    Vec4T<float> accum[kMaxElemPerThread];
    int L = end - start;
    for (int r = 0; r < L; r += kWarpSize) {
      int l = r + lane_id < L ? start + r + lane_id : 0;

      for (int j = 0; j < kWarpSize && r + j < L; ++j) {
        int j_ev = __shfl_sync(0xFFFFFFFF, l, j);
        const src_type *src_ev = copy_desc.get_src_ptr(j_ev);
        const float weight = copy_desc.get_sp_weight(j_ev);

#pragma unroll kMaxElemPerThread
        for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length;
             ++i) {
          Vec4T<src_type> src_elem;
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(vec_length - idx4, copy_width);
          src_elem.load(src_ev + idx4, n);
          accum[i].accumulate_multiply(src_elem, weight);
        }
      }
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i].val.x /= average_pooling_factor;
      accum[i].val.y /= average_pooling_factor;
      accum[i].val.z /= average_pooling_factor;
      accum[i].val.w /= average_pooling_factor;
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].store(dst_ev + idx4, n);
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void multi_to_one_weight_warp_per_ev_vec4_network_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int i_ev = blockIdx.x * blockDim.y + warp_id;
  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    float average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);

    dst_type *dst_ev = copy_desc.get_dst_ptr(i_ev);

    Vec4T<float> accum[kMaxElemPerThread];
    int L = end - start;
    for (int r = 0; r < L; r += kWarpSize) {
      int l = r + lane_id < L ? start + r + lane_id : 0;

      for (int j = 0; j < kWarpSize && r + j < L; ++j) {
        int j_ev = __shfl_sync(0xFFFFFFFF, l, j);
        const src_type *src_ev = copy_desc.get_src_ptr(j_ev);

#pragma unroll kMaxElemPerThread
        for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length;
             ++i) {
          Vec4T<src_type> src_elem;
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(vec_length - idx4, copy_width);
          src_elem.load(src_ev + idx4, n);
          accum[i].accumulate(src_elem);
        }
      }
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i].val.x /= average_pooling_factor;
      accum[i].val.y /= average_pooling_factor;
      accum[i].val.z /= average_pooling_factor;
      accum[i].val.w /= average_pooling_factor;
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].store(dst_ev + idx4, n);
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void multi_to_one_weight_warp_per_ev_vec4_model_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int i_ev = blockIdx.x * blockDim.y + warp_id;
  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);

    dst_type *dst_ev = copy_desc.get_dst_ptr(i_ev);

    Vec4T<float> accum[kMaxElemPerThread];
    int L = end - start;
    for (int r = 0; r < L; r += kWarpSize) {
      int l = r + lane_id < L ? start + r + lane_id : 0;

      for (int j = 0; j < kWarpSize && r + j < L; ++j) {
        int j_ev = __shfl_sync(0xFFFFFFFF, l, j);
        const src_type *src_ev = copy_desc.get_src_ptr(j_ev);
        const float weight = copy_desc.get_sp_weight(j_ev);

#pragma unroll kMaxElemPerThread
        for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length;
             ++i) {
          Vec4T<src_type> src_elem;
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(vec_length - idx4, copy_width);
          src_elem.load(src_ev + idx4, n);
          accum[i].accumulate_multiply(src_elem, weight);
        }
      }
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].store(dst_ev + idx4, n);
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void one_to_multi_cta_per_ev_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;
  int i_ev = blockIdx.x;

  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    int average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);
    const src_type *src_ev = copy_desc.get_src_ptr(i_ev);
    float accum[kMaxElemPerThread] = {0.f};
#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < vec_length; ++i) {
      accum[i] = src_ev[blockDim.x * i + threadIdx.x];
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i] /= average_pooling_factor;
    }

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);
    for (int r = 0; r < (end - start); ++r) {
      dst_type *dst_ev = copy_desc.get_dst_ptr(r + start);
#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < vec_length; ++i) {
        dst_ev[blockDim.x * i + threadIdx.x] =
            HugeCTR::TypeConvertFunc<dst_type, float>::convert(accum[i]);
      }
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void one_to_multi_weight_cta_per_ev_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;
  int i_ev = blockIdx.x;

  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    float average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);
    const src_type *src_ev = copy_desc.get_src_ptr(i_ev);
    float accum[kMaxElemPerThread] = {0.f};
#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < vec_length; ++i) {
      accum[i] = src_ev[blockDim.x * i + threadIdx.x];
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i] /= average_pooling_factor;
    }

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);
    for (int r = 0; r < (end - start); ++r) {
      dst_type *dst_ev = copy_desc.get_dst_ptr(r + start);
#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < vec_length; ++i) {
        dst_ev[blockDim.x * i + threadIdx.x] =
            HugeCTR::TypeConvertFunc<dst_type, float>::convert(accum[i]);
      }
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void one_to_multi_warp_per_ev_vec4_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int i_ev = blockIdx.x * blockDim.y + warp_id;
  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    int average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);
    const src_type *src_ev = copy_desc.get_src_ptr(i_ev);
    Vec4T<float> accum[kMaxElemPerThread];

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].load(src_ev + idx4, n);
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i].val.x /= average_pooling_factor;
      accum[i].val.y /= average_pooling_factor;
      accum[i].val.z /= average_pooling_factor;
      accum[i].val.w /= average_pooling_factor;
    }

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);
    int L = end - start;
    for (int r = 0; r < L; r += kWarpSize) {
      int l = r + lane_id < L ? start + r + lane_id : 0;

      for (int j = 0; j < kWarpSize && r + j < L; ++j) {
        int j_ev = __shfl_sync(0xFFFFFFFF, l, j);
        dst_type *dst_ev = copy_desc.get_dst_ptr(j_ev);

#pragma unroll kMaxElemPerThread
        for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length;
             ++i) {
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(vec_length - idx4, copy_width);
          accum[i].store(dst_ev + idx4, n);
        }
      }
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void one_to_multi_warp_per_ev_vec4_less_block_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;

  for (int i_ev = blockIdx.x * blockDim.y + warp_id; i_ev < copy_desc.num_vec_;
       i_ev += gridDim.x * blockDim.y) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    int average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);
    const src_type *src_ev = copy_desc.get_src_ptr(i_ev);
    Vec4T<float> accum[kMaxElemPerThread];

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].load(src_ev + idx4, n);
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i].val.x /= average_pooling_factor;
      accum[i].val.y /= average_pooling_factor;
      accum[i].val.z /= average_pooling_factor;
      accum[i].val.w /= average_pooling_factor;
    }

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);
    int L = end - start;
    for (int r = 0; r < L; ++r) {
      dst_type *dst_ev = copy_desc.get_dst_ptr(start + r);

#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min(vec_length - idx4, copy_width);
        accum[i].store(dst_ev + idx4, n);
      }
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void one_to_multi_weight_warp_per_ev_vec4_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int i_ev = blockIdx.x * blockDim.y + warp_id;
  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    float average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);
    const src_type *src_ev = copy_desc.get_src_ptr(i_ev);
    Vec4T<float> accum[kMaxElemPerThread];

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].load(src_ev + idx4, n);
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread; ++i) {
      accum[i].val.x /= average_pooling_factor;
      accum[i].val.y /= average_pooling_factor;
      accum[i].val.z /= average_pooling_factor;
      accum[i].val.w /= average_pooling_factor;
    }

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);
    int L = end - start;
    for (int r = 0; r < L; r += kWarpSize) {
      int l = r + lane_id < L ? start + r + lane_id : 0;

      for (int j = 0; j < kWarpSize && r + j < L; ++j) {
        int j_ev = __shfl_sync(0xFFFFFFFF, l, j);
        dst_type *dst_ev = copy_desc.get_dst_ptr(j_ev);

#pragma unroll kMaxElemPerThread
        for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length;
             ++i) {
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(vec_length - idx4, copy_width);
          accum[i].store(dst_ev + idx4, n);
        }
      }
    }
  }
}

template <typename SrcType, typename DstType, typename LambdaOffset, typename LambdaAverage,
          typename LambdaVecLength, typename LambdaSrcTensor, typename LambdaDstTensor>
struct MultiToOne {
  using SrcT = SrcType;
  using DstT = DstType;

  HOST_DEVICE_INLINE int get_offset(int i) { return static_cast<int>(get_offset_(i)); }
  HOST_DEVICE_INLINE int get_vec_length(int i) { return get_vec_length_(i); }
  HOST_DEVICE_INLINE int get_average_pooling_factor(int i) {
    return get_average_pooling_factor_(i);
  }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) { return get_src_tensor_(i); }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) { return get_dst_tensor_(i); }

  int num_vec_;
  LambdaOffset get_offset_;
  LambdaAverage get_average_pooling_factor_;
  LambdaVecLength get_vec_length_;
  LambdaSrcTensor get_src_tensor_;
  LambdaDstTensor get_dst_tensor_;
};

template <typename SrcType, typename DstType, typename LambdaOffset, typename LambdaAverage,
          typename LambdaVecLength, typename LambdaSrcTensor, typename LambdaDstTensor>
MultiToOne<SrcType, DstType, LambdaOffset, LambdaAverage, LambdaVecLength, LambdaSrcTensor,
           LambdaDstTensor>
make_MultiToOne(int num_vec, LambdaOffset get_offset, LambdaAverage get_average_pooling_factor,
                LambdaVecLength get_vec_length, LambdaSrcTensor get_src_tensor,
                LambdaDstTensor get_dst_tensor) {
  return {num_vec,        get_offset,     get_average_pooling_factor,
          get_vec_length, get_src_tensor, get_dst_tensor};
};
/*
template <typename SrcType, typename DstType, typename LambdaOffset, typename LambdaAverage,typename
LambdaHaveAverage, typename LambdaVecLength, typename LambdaSrcTensor, typename LambdaDstTensor,
typename LambdaHaveSpWeight, typename LambdaSpWeight> struct MultiToOneWeight { using SrcT =
SrcType; using DstT = DstType;

  HOST_DEVICE_INLINE int get_offset(int i) { return static_cast<int>(get_offset_(i)); }
  HOST_DEVICE_INLINE int get_vec_length(int i) { return get_vec_length_(i); }
  HOST_DEVICE_INLINE float get_average_pooling_factor(int i) {
    return get_average_pooling_factor_(i);
  }
  HOST_DEVICE_INLINE bool get_average_pooling_factor() {return have_average_pooling_();}
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) { return get_src_tensor_(i); }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) { return get_dst_tensor_(i); }
  HOST_DEVICE_INLINE bool have_sp_weight() { return have_sp_weight_(); }
  HOST_DEVICE_INLINE float get_sp_weight(int i) { return get_sp_weight_(i); }

  int num_vec_;
  LambdaOffset get_offset_;
  LambdaAverage get_average_pooling_factor_;
  LambdaHaveAverage have_average_pooling_;
  LambdaVecLength get_vec_length_;
  LambdaSrcTensor get_src_tensor_;
  LambdaDstTensor get_dst_tensor_;
  LambdaHaveSpWeight have_sp_weight_;
  LambdaSpWeight get_sp_weight_;
};
*/
template <typename SrcType, typename DstType, typename LambdaOffset, typename LambdaAverage,
          typename LambdaVecLength, typename LambdaSrcTensor, typename LambdaDstTensor,
          typename LambdaSpWeight>
struct MultiToOneWeight {
  using SrcT = SrcType;
  using DstT = DstType;

  HOST_DEVICE_INLINE int get_offset(int i) { return static_cast<int>(get_offset_(i)); }
  HOST_DEVICE_INLINE int get_vec_length(int i) { return get_vec_length_(i); }
  HOST_DEVICE_INLINE float get_average_pooling_factor(int i) {
    return get_average_pooling_factor_(i);
  }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) { return get_src_tensor_(i); }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) { return get_dst_tensor_(i); }
  HOST_DEVICE_INLINE float get_sp_weight(int i) { return get_sp_weight_(i); }

  int num_vec_;
  LambdaOffset get_offset_;
  LambdaAverage get_average_pooling_factor_;
  LambdaVecLength get_vec_length_;
  LambdaSrcTensor get_src_tensor_;
  LambdaDstTensor get_dst_tensor_;
  LambdaSpWeight get_sp_weight_;
};

template <typename SrcType, typename DstType, typename LambdaOffset, typename LambdaAverage,
          typename LambdaVecLength, typename LambdaSrcTensor, typename LambdaDstTensor,
          typename LambdaSpWeight>
MultiToOneWeight<SrcType, DstType, LambdaOffset, LambdaAverage, LambdaVecLength, LambdaSrcTensor,
                 LambdaDstTensor, LambdaSpWeight>
make_MultiToOneWeight(int num_vec, LambdaOffset get_offset,
                      LambdaAverage get_average_pooling_factor, LambdaVecLength get_vec_length,
                      LambdaSrcTensor get_src_tensor, LambdaDstTensor get_dst_tensor,
                      LambdaSpWeight get_sp_weight) {
  return {num_vec,        get_offset,   get_average_pooling_factor, get_vec_length, get_src_tensor,
          get_dst_tensor, get_sp_weight};
};

template <typename SrcType, typename DstType, typename LambdaVecNum, typename LambdaSrcVecLength,
          typename LambdaDstId, typename LambdaSrcPtr, typename LambdaDstPtr>
struct MultiToOne_reduce_new {
  using SrcT = SrcType;
  using DstT = DstType;

  HOST_DEVICE_INLINE size_t num_vec() { return num_vec_(); }
  HOST_DEVICE_INLINE int get_src_vec_length(int i) { return get_src_vec_length_(i); }
  HOST_DEVICE_INLINE uint32_t get_dst_id(int i) { return get_dst_id_(i); }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) { return get_src_ptr_(i); }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) { return get_dst_ptr_(i); }

  LambdaVecNum num_vec_;
  LambdaSrcVecLength get_src_vec_length_;
  LambdaDstId get_dst_id_;
  LambdaSrcPtr get_src_ptr_;
  LambdaDstPtr get_dst_ptr_;
};

template <typename SrcType, typename DstType, typename LambdaVecNum, typename LambdaSrcVecLength,
          typename LambdaDstId, typename LambdaSrcPtr, typename LambdaDstPtr>
MultiToOne_reduce_new<SrcType, DstType, LambdaVecNum, LambdaSrcVecLength, LambdaDstId, LambdaSrcPtr,
                      LambdaDstPtr>
make_MultiToOne_reduce_new(LambdaVecNum num_vec, LambdaSrcVecLength get_src_vec_length,
                           LambdaDstId get_dst_id, LambdaSrcPtr get_src_ptr,
                           LambdaDstPtr get_dst_ptr) {
  return {num_vec, get_src_vec_length, get_dst_id, get_src_ptr, get_dst_ptr};
};

template <typename SrcType, typename DstType, typename LambdaKey, typename LambdaSrcVecLength,
          typename LambdaDstVecLength, typename LambdaDstUniqueId, typename LambdaSrcTensor,
          typename LambdaDstTensor>

struct MultiToOne_reduce {
  using SrcT = SrcType;
  using DstT = DstType;

  HOST_DEVICE_INLINE uint32_t get_key(int i) { return get_key_(i); }
  HOST_DEVICE_INLINE int get_src_vec_length(int i) { return get_src_vec_length_(i); }
  HOST_DEVICE_INLINE int get_dst_vec_length(int i) { return get_dst_vec_length_(i); }
  HOST_DEVICE_INLINE uint32_t get_dst_unique_id(int i) { return get_dst_unique_id_(i); }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) { return get_src_tensor_(i); }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) { return get_dst_tensor_(i); }

  int num_vec_;
  LambdaKey get_key_;
  LambdaSrcVecLength get_src_vec_length_;
  LambdaDstVecLength get_dst_vec_length_;
  LambdaDstUniqueId get_dst_unique_id_;
  LambdaSrcTensor get_src_tensor_;
  LambdaDstTensor get_dst_tensor_;
};

template <typename SrcType, typename DstType, typename LambdaKey, typename LambdaSrcVecLength,
          typename LambdaDstVecLength, typename LambdaDstUniqueId, typename LambdaSrcTensor,
          typename LambdaDstTensor>
MultiToOne_reduce<SrcType, DstType, LambdaKey, LambdaSrcVecLength, LambdaDstVecLength,
                  LambdaDstUniqueId, LambdaSrcTensor, LambdaDstTensor>
make_MultiToOne_reduce(int num_vec, LambdaKey get_key, LambdaSrcVecLength get_src_vec_length,
                       LambdaDstVecLength get_dst_vec_length, LambdaDstUniqueId get_dst_unique_id,
                       LambdaSrcTensor get_src_tensor, LambdaDstTensor get_dst_tensor) {
  return {num_vec,           get_key,        get_src_vec_length, get_dst_vec_length,
          get_dst_unique_id, get_src_tensor, get_dst_tensor};
};

template <typename SrcType, typename DstType, typename LambdaKey, typename LambdaSrcVecLength,
          typename LambdaDstVecLength, typename LambdaDstUniqueId, typename LambdaSrcTensor,
          typename LambdaDstTensor, typename LambdaWeightTensor>
struct MultiToOne_reduce_weight {
  using SrcT = SrcType;
  using DstT = DstType;

  HOST_DEVICE_INLINE uint32_t get_key(int i) { return get_key_(i); }
  HOST_DEVICE_INLINE int get_src_vec_length(int i) { return get_src_vec_length_(i); }
  HOST_DEVICE_INLINE int get_dst_vec_length(int i) { return get_dst_vec_length_(i); }
  HOST_DEVICE_INLINE uint32_t get_dst_unique_id(int i) { return get_dst_unique_id_(i); }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) { return get_src_tensor_(i); }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) { return get_dst_tensor_(i); }
  HOST_DEVICE_INLINE float get_weight(int i) { return get_weight_(i); }

  int num_vec_;
  LambdaKey get_key_;
  LambdaSrcVecLength get_src_vec_length_;
  LambdaDstVecLength get_dst_vec_length_;
  LambdaDstUniqueId get_dst_unique_id_;
  LambdaSrcTensor get_src_tensor_;
  LambdaDstTensor get_dst_tensor_;
  LambdaWeightTensor get_weight_;
};

template <typename SrcType, typename DstType, typename LambdaKey, typename LambdaSrcVecLength,
          typename LambdaDstVecLength, typename LambdaDstUniqueId, typename LambdaSrcTensor,
          typename LambdaDstTensor, typename LambdaWeightTensor>
MultiToOne_reduce_weight<SrcType, DstType, LambdaKey, LambdaSrcVecLength, LambdaDstVecLength,
                         LambdaDstUniqueId, LambdaSrcTensor, LambdaDstTensor, LambdaWeightTensor>
make_MultiToOne_reduce_weight(int num_vec, LambdaKey get_key, LambdaSrcVecLength get_src_vec_length,
                              LambdaDstVecLength get_dst_vec_length,
                              LambdaDstUniqueId get_dst_unique_id, LambdaSrcTensor get_src_tensor,
                              LambdaDstTensor get_dst_tensor, LambdaWeightTensor get_weight) {
  return {num_vec,           get_key,        get_src_vec_length, get_dst_vec_length,
          get_dst_unique_id, get_src_tensor, get_dst_tensor,     get_weight};
};

template <typename CopyDesc>
void copy_multi_to_one(CopyDesc copy_desc, int max_ev_size, cudaStream_t stream) {
  if (max_ev_size <= 128) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 2};
    multi_to_one_warp_per_ev_vec4_kernel<CopyDesc, 1>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 256) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 2};
    multi_to_one_warp_per_ev_vec4_kernel<CopyDesc, 2>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 1024) {
    multi_to_one_cta_per_ev_kernel<CopyDesc, 1>
        <<<copy_desc.num_vec_, max_ev_size, 0, stream>>>(copy_desc);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "HugeCTR does not support emb vector size >= 4096");
  }
}

template <typename CopyDesc>
void copy_multi_to_one(CopyDesc copy_desc, const HugeCTR::core23::KernelParams &kernel_params,
                       int max_ev_size, cudaStream_t stream) {
  if (max_ev_size <= 128) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 8};
    int num_vector_per_warp = NUM_VECTOR_PER_WARP;
    get_kernel_config_use_warp(kernel_params.num_sms, kernel_params.max_thread_per_sm, 256,
                               kernel_params.warp_size, copy_desc.num_vec_, &grid_size,
                               &num_vector_per_warp, 8);
    multi_to_one_warp_per_ev_vec4_less_block_kernel<CopyDesc, 1>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 256) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 8};
    int num_vector_per_warp = NUM_VECTOR_PER_WARP;
    get_kernel_config_use_warp(kernel_params.num_sms, kernel_params.max_thread_per_sm, 256,
                               kernel_params.warp_size, copy_desc.num_vec_, &grid_size,
                               &num_vector_per_warp, 8);

    multi_to_one_warp_per_ev_vec4_less_block_kernel<CopyDesc, 2>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 1024) {
    int grid_size = copy_desc.num_vec_;

    multi_to_one_cta_per_ev_kernel<CopyDesc, 1>
        <<<copy_desc.num_vec_, max_ev_size, 0, stream>>>(copy_desc);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "HugeCTR does not support emb vector size >= 4096");
  }
}

template <typename CopyDesc>
void copy_multi_to_one_weight(CopyDesc copy_desc, int max_ev_size, cudaStream_t stream) {
  if (max_ev_size <= 128) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 2};
    multi_to_one_weight_warp_per_ev_vec4_kernel<CopyDesc, 1>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 256) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 2};
    multi_to_one_weight_warp_per_ev_vec4_kernel<CopyDesc, 2>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 1024) {
    multi_to_one_weight_cta_per_ev_kernel<CopyDesc, 1>
        <<<copy_desc.num_vec_, max_ev_size, 0, stream>>>(copy_desc);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "HugeCTR does not support emb vector size >= 4096");
  }
}

template <typename CopyDesc>
void copy_one_to_multi(CopyDesc copy_desc, int max_ev_size, cudaStream_t stream) {
  if (max_ev_size <= 128) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 2};
    one_to_multi_warp_per_ev_vec4_kernel<CopyDesc, 1>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 256) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 2};
    one_to_multi_warp_per_ev_vec4_kernel<CopyDesc, 2>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 1024) {
    one_to_multi_cta_per_ev_kernel<CopyDesc, 1>
        <<<copy_desc.num_vec_, max_ev_size, 0, stream>>>(copy_desc);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "HugeCTR does not support emb vector size >= 4096");
  }
}

template <typename CopyDesc>
void copy_one_to_multi(CopyDesc copy_desc, const HugeCTR::core23::KernelParams &kernel_params,
                       int max_ev_size, cudaStream_t stream) {
  if (max_ev_size <= 128) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 8};
    int num_vector_per_warp = NUM_VECTOR_PER_WARP;
    get_kernel_config_use_warp(kernel_params.num_sms, kernel_params.max_thread_per_sm, 256,
                               kernel_params.warp_size, copy_desc.num_vec_, &grid_size,
                               &num_vector_per_warp, 8);

    one_to_multi_warp_per_ev_vec4_less_block_kernel<CopyDesc, 1>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 256) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 8};
    int num_vector_per_warp = NUM_VECTOR_PER_WARP;
    get_kernel_config_use_warp(kernel_params.num_sms, kernel_params.max_thread_per_sm, 256,
                               kernel_params.warp_size, copy_desc.num_vec_, &grid_size,
                               &num_vector_per_warp, 8);

    one_to_multi_warp_per_ev_vec4_less_block_kernel<CopyDesc, 2>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 1024) {
    int grid_size = copy_desc.num_vec_;

    one_to_multi_cta_per_ev_kernel<CopyDesc, 1>
        <<<copy_desc.num_vec_, max_ev_size, 0, stream>>>(copy_desc);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "HugeCTR does not support emb vector size >= 4096");
  }
}

template <typename CopyDesc>
void copy_one_to_multi_weight(CopyDesc copy_desc, int max_ev_size, cudaStream_t stream) {
  if (max_ev_size <= 128) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 2};
    one_to_multi_weight_warp_per_ev_vec4_kernel<CopyDesc, 1>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 256) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 2};
    one_to_multi_weight_warp_per_ev_vec4_kernel<CopyDesc, 2>
        <<<grid_size, block_size, 0, stream>>>(copy_desc);
  } else if (max_ev_size <= 1024) {
    one_to_multi_weight_cta_per_ev_kernel<CopyDesc, 1>
        <<<copy_desc.num_vec_, max_ev_size, 0, stream>>>(copy_desc);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "HugeCTR does not support emb vector size >= 4096");
  }
}

}  // namespace embedding

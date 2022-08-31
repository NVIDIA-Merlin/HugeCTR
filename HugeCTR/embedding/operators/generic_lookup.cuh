/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "../view.hpp"
#include "HugeCTR/include/utils.cuh"

namespace embedding {

template <typename T>
struct Vec4T {};

template <>
struct Vec4T<__half> {
  __half2 first;
  __half2 second;

  DEVICE_INLINE Vec4T() {
    first.x = 0.f;
    first.y = 0.f;
    second.x = 0.f;
    second.y = 0.f;
  }

  DEVICE_INLINE void load(const float *p, int n) {
    if (n == 4) {
      float4 f = *(reinterpret_cast<const float4 *>(p));
      float2 firstf{f.x, f.y};
      float2 secondf{f.z, f.w};
      first = __float22half2_rn(firstf);
      second = __float22half2_rn(secondf);
    } else {
      if (n > 0) first.x = __float2half(p[0]);
      if (n > 1) first.y = __float2half(p[1]);
      if (n > 2) second.x = __float2half(p[2]);
    }
  }

  DEVICE_INLINE void load(const __half *p, int n) {
    if (n == 4) {
      first = *(reinterpret_cast<const __half2 *>(p));
      second = *(reinterpret_cast<const __half2 *>(p + 2));
    } else {
      if (n > 0) first.x = p[0];
      if (n > 1) first.y = p[1];
      if (n > 2) second.x = p[2];
    }
  }

  DEVICE_INLINE void store(float *dst, int n) {
    if (n == 4) {
      float4 f;
      f.x = __half2float(first.x);
      f.y = __half2float(first.y);
      f.z = __half2float(second.x);
      f.w = __half2float(second.y);
      *(reinterpret_cast<float4 *>(dst)) = f;
    } else {
      if (n > 0) dst[0] = __half2float(first.x);
      if (n > 1) dst[1] = __half2float(first.y);
      if (n > 2) dst[2] = __half2float(second.x);
    }
  }

  DEVICE_INLINE void store(__half *dst, int n) {
    if (n == 4) {
      *(reinterpret_cast<__half2 *>(dst)) = first;
      *(reinterpret_cast<__half2 *>(dst + 2)) = second;
    } else {
      if (n > 0) dst[0] = first.x;
      if (n > 1) dst[1] = first.y;
      if (n > 2) dst[2] = second.x;
    }
  }

  DEVICE_INLINE void atomic_store_accum(float *dst, int n) {
    if (n > 0) atomicAdd(dst, __half2float(first.x));
    if (n > 1) atomicAdd(dst + 1, __half2float(first.y));
    if (n > 2) atomicAdd(dst + 2, __half2float(second.x));
    if (n > 3) atomicAdd(dst + 3, __half2float(second.y));
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
      val.x = __half2float(h.first.x);
      val.y = __half2float(h.first.y);
      val.z = __half2float(h.second.x);
      val.w = __half2float(h.second.y);
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

  DEVICE_INLINE void accumulate(const Vec4T<float> &other) {
    val.x += other.val.x;
    val.y += other.val.y;
    val.z += other.val.z;
    val.w += other.val.w;
  }

  DEVICE_INLINE void accumulate(const Vec4T<__half> &other) {
    val.x += __half2float(other.first.x);
    val.y += __half2float(other.first.y);
    val.z += __half2float(other.second.x);
    val.w += __half2float(other.second.y);
  }
};

template <typename IndexArray, typename OffsetArray, typename DstArray, typename SrcTensor,
          typename DstTensor, int kMaxElemPerThread>
__global__ void generic_lookup_cta_per_bucket_kernel(IndexArray idx, OffsetArray offset_idx,
                                                     DstArray dst_idx, SrcTensor src_tensor,
                                                     DstTensor dst_tensor) {
  using index_t = typename IndexArray::value_type;
  using offset_t = typename OffsetArray::value_type;
  using dst_index_t = typename DstArray::value_type;
  using src_t = typename SrcTensor::value_type;
  using src_scalar_t = typename SrcTensor::value_type::value_type;
  using dst_t = typename DstTensor::value_type;
  using dst_scalar_t = typename DstTensor::value_type::value_type;

  int bucket_index = blockIdx.x;
  if (bucket_index < offset_idx.size() - 1) {
    offset_t start = offset_idx[bucket_index];
    offset_t end = offset_idx[bucket_index + 1];

    float accum[kMaxElemPerThread] = {0.f};
    for (int r = 0; r < static_cast<int>(end) - static_cast<int>(start); ++r) {
      index_t in_bucket_id = idx[start + r];
      src_t ev = src_tensor[in_bucket_id];

#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < ev.size(); ++i) {
        accum[i] += HugeCTR::TypeConvertFunc<float, src_scalar_t>::convert(
            ev[blockDim.x * i + threadIdx.x]);
      }
    }

    dst_index_t out_bucket_id = dst_idx[bucket_index];
    dst_t dst_ev = dst_tensor[out_bucket_id];

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < dst_ev.size(); ++i) {
      dst_ev[blockDim.x * i + threadIdx.x] =
          HugeCTR::TypeConvertFunc<dst_scalar_t, float>::convert(accum[i]);
    }
  }
}

template <typename IndexArray, typename OffsetArray, typename DstArray, typename SrcTensor,
          typename DstTensor, int kMaxElemPerThread>
__global__ void generic_lookup_cta_per_bucket_vec4_kernel(IndexArray idx, OffsetArray offset_idx,
                                                          DstArray dst_idx, SrcTensor src_tensor,
                                                          DstTensor dst_tensor) {
  using index_t = typename IndexArray::value_type;
  using offset_t = typename OffsetArray::value_type;
  using dst_index_t = typename DstArray::value_type;
  using src_t = typename SrcTensor::value_type;
  using src_scalar_t = typename SrcTensor::value_type::value_type;
  using dst_t = typename DstTensor::value_type;
  using dst_scalar_t = typename DstTensor::value_type::value_type;
  constexpr int copy_width = 4;

  int bucket_index = blockIdx.x;
  if (bucket_index < offset_idx.size() - 1) {
    offset_t start = offset_idx[bucket_index];
    offset_t end = offset_idx[bucket_index + 1];

    Vec4T<float> accum[kMaxElemPerThread];
    for (int r = 0; r < static_cast<int>(end) - static_cast<int>(start); ++r) {
      index_t in_bucket_id = idx[start + r];
      src_t ev = src_tensor[in_bucket_id];

#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread && 4 * blockDim.x * i + 4 * threadIdx.x < ev.size();
           ++i) {
        Vec4T<src_scalar_t> src_elem;
        int idx4 = 4 * blockDim.x * i + 4 * threadIdx.x;
        int n = min(ev.size() - idx4, copy_width);
        src_elem.load(&ev[idx4], n);
        accum[i].accumulate(src_elem);
      }
    }

    dst_index_t out_bucket_id = dst_idx[bucket_index];
    dst_t dst_ev = dst_tensor[out_bucket_id];

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && 4 * blockDim.x * i + 4 * threadIdx.x < dst_ev.size();
         ++i) {
      int idx4 = 4 * blockDim.x * i + 4 * threadIdx.x;
      int n = min(dst_ev.size() - idx4, copy_width);
      accum[i].store(&dst_ev[idx4], n);
    }
  }
}

template <typename IndexArray, typename OffsetArray, typename DstArray, typename SrcTensor,
          typename DstTensor, int kMaxElemPerThread>
__global__ void generic_lookup_warp_per_bucket_vec4_kernel(IndexArray idx, OffsetArray offset_idx,
                                                           DstArray dst_idx, SrcTensor src_tensor,
                                                           DstTensor dst_tensor) {
  using index_t = typename IndexArray::value_type;
  using offset_t = typename OffsetArray::value_type;
  using dst_index_t = typename DstArray::value_type;
  using src_t = typename SrcTensor::value_type;
  using src_scalar_t = typename SrcTensor::value_type::value_type;
  using dst_t = typename DstTensor::value_type;
  using dst_scalar_t = typename DstTensor::value_type::value_type;
  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int bucket_index = blockIdx.x * blockDim.y + warp_id;
  if (bucket_index < offset_idx.size() - 1) {
    offset_t start = offset_idx[bucket_index];
    offset_t end = offset_idx[bucket_index + 1];

    Vec4T<float> accum[kMaxElemPerThread];
    int L = static_cast<int>(end) - static_cast<int>(start);
    for (int r = 0; r < L; r += kWarpSize) {
      int l = r + lane_id;
      index_t in_bucket_id = l < L ? idx[start + l] : 0;

      for (int j = 0; j < kWarpSize && r + j < L; ++j) {
        int in_bid = __shfl_sync(0xFFFFFFFF, in_bucket_id, j);
        src_t ev = src_tensor[in_bid];

#pragma unroll kMaxElemPerThread
        for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < ev.size(); ++i) {
          Vec4T<src_scalar_t> src_elem;
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(ev.size() - idx4, copy_width);
          src_elem.load(&ev[idx4], n);
          accum[i].accumulate(src_elem);
        }
      }
    }

    dst_index_t out_bucket_id = dst_idx[bucket_index];
    dst_t dst_ev = dst_tensor[out_bucket_id];

#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < dst_ev.size(); ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(dst_ev.size() - idx4, copy_width);
      accum[i].store(&dst_ev[idx4], n);
    }
  }
}

template <typename IndexArray, typename OffsetArray, typename DstArray, typename SrcTensor,
          typename DstTensor>
void generic_lookup(IndexArray idx, OffsetArray offset_idx, DstArray dst_idx, SrcTensor src_tensor,
                    DstTensor dst_tensor, int max_ev_size, cudaStream_t stream) {
  int num_bucket = offset_idx.size() - 1;
  if (max_ev_size <= 64) {
    generic_lookup_cta_per_bucket_kernel<IndexArray, OffsetArray, DstArray, SrcTensor, DstTensor, 1>
        <<<num_bucket, max_ev_size, 0, stream>>>(idx, offset_idx, dst_idx, src_tensor, dst_tensor);
  } else if (max_ev_size <= 128) {
    int grid_size = (num_bucket - 1) / 2 + 1;
    dim3 block_size{32, 2};
    generic_lookup_warp_per_bucket_vec4_kernel<IndexArray, OffsetArray, DstArray, SrcTensor,
                                               DstTensor, 1>
        <<<grid_size, block_size, 0, stream>>>(idx, offset_idx, dst_idx, src_tensor, dst_tensor);
  } else if (max_ev_size <= 1024) {
    generic_lookup_cta_per_bucket_vec4_kernel<IndexArray, OffsetArray, DstArray, SrcTensor,
                                              DstTensor, 1>
        <<<num_bucket, (max_ev_size - 1) / 4 + 1, 0, stream>>>(idx, offset_idx, dst_idx, src_tensor,
                                                               dst_tensor);
  } else if (max_ev_size <= 2048) {
    generic_lookup_cta_per_bucket_vec4_kernel<IndexArray, OffsetArray, DstArray, SrcTensor,
                                              DstTensor, 2>
        <<<num_bucket, 256, 0, stream>>>(idx, offset_idx, dst_idx, src_tensor, dst_tensor);
  } else if (max_ev_size <= 4096) {
    generic_lookup_cta_per_bucket_vec4_kernel<IndexArray, OffsetArray, DstArray, SrcTensor,
                                              DstTensor, 4>
        <<<num_bucket, 256, 0, stream>>>(idx, offset_idx, dst_idx, src_tensor, dst_tensor);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "HugeCTR does not support emb vector size >= 4096");
  }
}

template <typename IndexArray, typename DstArray, typename SrcTensor, typename DstTensor,
          int kMaxElemPerThread>
__global__ void accumulate_grad_warp_per_bucket_vec4_atomic_kernel(IndexArray src_idx,
                                                                   DstArray dst_idx,
                                                                   const uint32_t *num_idx,
                                                                   SrcTensor src_tensor,
                                                                   DstTensor dst_tensor) {
  using index_t = typename IndexArray::value_type;
  using dst_index_t = typename DstArray::value_type;
  using src_t = typename SrcTensor::value_type;
  using src_scalar_t = typename SrcTensor::value_type::value_type;
  using dst_t = typename DstTensor::value_type;
  using dst_scalar_t = typename DstTensor::value_type::value_type;
  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;
  int kWarpNum = blockDim.y * gridDim.x;

  int lane_id = threadIdx.x;
  int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  for (int i = 0; i * kWarpNum + warp_id < *num_idx; ++i) {
    int idx = i * kWarpNum + warp_id;
    index_t s_idx = src_idx[idx];
    src_t s_ev = src_tensor[s_idx];

    dst_index_t d_idx = dst_idx[idx];
    dst_t d_ev = dst_tensor[d_idx];

    Vec4T<src_scalar_t> s_vec[kMaxElemPerThread];
#pragma unroll kMaxElemPerThread
    for (int j = 0; j < kMaxElemPerThread && 4 * kWarpSize * j + 4 * lane_id < s_ev.size(); ++j) {
      int idx4 = 4 * kWarpSize * j + 4 * lane_id;
      int n = min(s_ev.size() - idx4, copy_width);
      s_vec[j].load(&s_ev[idx4], n);

      s_vec[j].atomic_store_accum(&d_ev[idx4], n);
    }
  }
}

template <typename IndexArray, typename DstArray, typename SrcTensor, typename DstTensor>
void accumulate_grad(IndexArray src_idx, DstArray dst_idx, const uint32_t *num_idx,
                     SrcTensor src_tensor, DstTensor dst_tensor, int num_unique_idx,
                     int max_ev_size, cudaStream_t stream) {
  if (max_ev_size <= 128) {
    int grid_size = (num_unique_idx - 1) / 2 + 1;
    dim3 block_size{32, 2};
    accumulate_grad_warp_per_bucket_vec4_atomic_kernel<IndexArray, DstArray, SrcTensor, DstTensor,
                                                       1>
        <<<grid_size, block_size, 0, stream>>>(src_idx, dst_idx, num_idx, src_tensor, dst_tensor);
  } else if (max_ev_size <= 256) {
    int grid_size = (num_unique_idx - 1) / 2 + 1;
    dim3 block_size{32, 2};
    accumulate_grad_warp_per_bucket_vec4_atomic_kernel<IndexArray, DstArray, SrcTensor, DstTensor,
                                                       2>
        <<<grid_size, block_size, 0, stream>>>(src_idx, dst_idx, num_idx, src_tensor, dst_tensor);
  } else if (max_ev_size <= 512) {
    int grid_size = (num_unique_idx - 1) / 2 + 1;
    dim3 block_size{32, 2};
    accumulate_grad_warp_per_bucket_vec4_atomic_kernel<IndexArray, DstArray, SrcTensor, DstTensor,
                                                       4>
        <<<grid_size, block_size, 0, stream>>>(src_idx, dst_idx, num_idx, src_tensor, dst_tensor);
  } else if (max_ev_size <= 1024) {
    int grid_size = (num_unique_idx - 1) / 2 + 1;
    dim3 block_size{32, 2};
    accumulate_grad_warp_per_bucket_vec4_atomic_kernel<IndexArray, DstArray, SrcTensor, DstTensor,
                                                       8>
        <<<grid_size, block_size, 0, stream>>>(src_idx, dst_idx, num_idx, src_tensor, dst_tensor);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "HugeCTR does not support emb vector size >= 1024");
  }
}

template <typename IndexArray, typename ScalerArray, typename DstArray, typename SrcTensor,
          typename DstTensor>
__global__ void generic_copy_cta_per_bucket_vec4_kernel(IndexArray src_idx, ScalerArray scaler_arr,
                                                        DstArray dst_idx, SrcTensor src_tensor,
                                                        DstTensor dst_tensor) {
  using index_t = typename IndexArray::value_type;
  using scaler_t = typename ScalerArray::value_type;
  using dst_index_t = typename DstArray::value_type;
  using src_t = typename SrcTensor::value_type;
  using src_scalar_t = typename SrcTensor::value_type::value_type;
  using dst_t = typename DstTensor::value_type;
  using dst_scalar_t = typename DstTensor::value_type::value_type;
  constexpr int copy_width = 4;
  int kThreadPerBlock = blockDim.x;
  int tid = threadIdx.x;
  int idx = blockIdx.x;

  if (idx < src_idx.size()) {
    index_t s_idx = src_idx[idx];
    src_t s_ev = src_tensor[s_idx];

    dst_index_t d_idx = dst_idx[idx];
    dst_t d_ev = dst_tensor[d_idx];

    float scaler = 1.f / static_cast<float>(scaler_arr[idx]);

    for (int i = 0; 4 * kThreadPerBlock * i + 4 * tid < s_ev.size(); ++i) {
      Vec4T<float> s_vec;
      int idx4 = 4 * kThreadPerBlock * i + 4 * tid;
      int n = min(s_ev.size() - idx4, copy_width);
      s_vec.load(&s_ev[idx4], n);

      s_vec.val.x *= scaler;
      s_vec.val.y *= scaler;
      s_vec.val.z *= scaler;
      s_vec.val.w *= scaler;

      s_vec.store(&d_ev[idx4], n);
    }
  }
}

template <typename IndexArray, typename ScalerArray, typename DstArray, typename SrcTensor,
          typename DstTensor>
void generic_copy(IndexArray src_idx, ScalerArray scaler_arr, DstArray dst_idx,
                  SrcTensor src_tensor, DstTensor dst_tensor, int max_ev_size,
                  cudaStream_t stream) {
  int num_idx = src_idx.size();
  generic_copy_cta_per_bucket_vec4_kernel<<<num_idx, (max_ev_size - 1) / 4 + 1, 0, stream>>>(
      src_idx, scaler_arr, dst_idx, src_tensor, dst_tensor);
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void multi_to_one_cta_per_ev_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;
  int i_ev = blockIdx.x;

  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length_(i_ev);
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
    vec_length_type vec_length = copy_desc.get_vec_length_(i_ev);
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
__global__ void one_to_multi_cta_per_ev_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;
  int i_ev = blockIdx.x;

  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length_(i_ev);
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
    vec_length_type vec_length = copy_desc.get_vec_length_(i_ev);
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

template <typename CopyDesc>
void copy_multi_to_one(CopyDesc copy_desc, int max_ev_size, cudaStream_t stream,
                       bool backward = false) {
  if (max_ev_size <= 128) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 2};
    if (!backward) {
      multi_to_one_warp_per_ev_vec4_kernel<CopyDesc, 1>
          <<<grid_size, block_size, 0, stream>>>(copy_desc);
    } else {
      one_to_multi_warp_per_ev_vec4_kernel<CopyDesc, 1>
          <<<grid_size, block_size, 0, stream>>>(copy_desc);
    }
  } else if (max_ev_size <= 256) {
    int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
    dim3 block_size{32, 2};
    if (!backward) {
      multi_to_one_warp_per_ev_vec4_kernel<CopyDesc, 2>
          <<<grid_size, block_size, 0, stream>>>(copy_desc);
    } else {
      one_to_multi_warp_per_ev_vec4_kernel<CopyDesc, 2>
          <<<grid_size, block_size, 0, stream>>>(copy_desc);
    }
  } else if (max_ev_size <= 1024) {
    if (!backward) {
      multi_to_one_cta_per_ev_kernel<CopyDesc, 1>
          <<<copy_desc.num_vec_, max_ev_size, 0, stream>>>(copy_desc);
    } else {
      one_to_multi_cta_per_ev_kernel<CopyDesc, 1>
          <<<copy_desc.num_vec_, max_ev_size, 0, stream>>>(copy_desc);
    }
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "HugeCTR does not support emb vector size >= 4096");
  }
}
}  // namespace embedding
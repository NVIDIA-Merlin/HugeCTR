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

template <typename IndexArray, typename OffsetArray, typename CountArray, typename DstArray,
          typename SrcTensor, typename DstTensor, int kMaxElemPerThread, int kThreadPerBucket,
          int TPB, bool debug = false,
          typename = typename std::enable_if_t<(kThreadPerBucket == TPB)>>
__global__ void generic_lookup_per_cta_kernel(IndexArray idx, OffsetArray offset_idx,
                                              CountArray count, DstArray dst_idx,
                                              SrcTensor src_tensor, DstTensor dst_tensor) {
  using index_t = typename IndexArray::value_type;
  using offset_t = typename OffsetArray::value_type;
  using count_t = typename CountArray::value_type;
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
      // if (debug) {
      //   printf("bucket_index:%d,start:%d,end:%d,in_bucket_id:%d\n", bucket_index, start, end,
      //   in_bucket_id);
      // }
#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread && kThreadPerBucket * i + threadIdx.x < ev.size();
           ++i) {
        if (debug) {
          printf("before accum:%f\n", HugeCTR::TypeConvertFunc<float, src_scalar_t>::convert(
                                          ev[kThreadPerBucket * i + threadIdx.x]));
        }
        accum[i] += HugeCTR::TypeConvertFunc<float, src_scalar_t>::convert(
            ev[kThreadPerBucket * i + threadIdx.x]);
      }
    }

    if (count[bucket_index] > 1) {
#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread; ++i) {
        accum[i] /= count[bucket_index];
      }
    }

    dst_index_t out_bucket_id = dst_idx[bucket_index];
    dst_t dst_ev = dst_tensor[out_bucket_id];
    // if (debug) {
    //   printf("bucket_index:%d,out_bucket_id:%d\n", bucket_index, out_bucket_id);
    // }
#pragma unroll kMaxElemPerThread
    for (int i = 0; i < kMaxElemPerThread && kThreadPerBucket * i + threadIdx.x < dst_ev.size();
         ++i) {
      if (debug) {
        printf("accum:%f\n", accum[i]);
      }
      dst_ev[i * kThreadPerBucket + threadIdx.x] =
          HugeCTR::TypeConvertFunc<dst_scalar_t, float>::convert(accum[i]);
    }
  }
}

template <typename IndexArray, typename OffsetArray, typename CountArray, typename DstArray,
          typename SrcTensor, typename DstTensor>
void generic_lookup(IndexArray idx, OffsetArray offset_idx, CountArray count, DstArray dst_idx,
                    SrcTensor src_tensor, DstTensor dst_tensor,
                    // int max_elem_per_thread, int thread_per_bucket,
                    cudaStream_t stream, bool debug = false) {
  int num_bucket = offset_idx.size() - 1;
  if (debug) {
    generic_lookup_per_cta_kernel<IndexArray, OffsetArray, CountArray, DstArray, SrcTensor,
                                  DstTensor, 4, 512, 512, true>
        <<<num_bucket, 512, 0, stream>>>(idx, offset_idx, count, dst_idx, src_tensor, dst_tensor);
  } else {
    generic_lookup_per_cta_kernel<IndexArray, OffsetArray, CountArray, DstArray, SrcTensor,
                                  DstTensor, 4, 512, 512>
        <<<num_bucket, 512, 0, stream>>>(idx, offset_idx, count, dst_idx, src_tensor, dst_tensor);
  }
}

}  // namespace embedding
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

#include <cassert>
#include <shuffle/configs.cuh>
#include <shuffle/descriptors.cuh>
#include <shuffle/vector_conversion.cuh>

namespace ShuffleKernels {

template <typename VecType, typename T>
__device__ inline void global2shmem_vec(T* shmem_dst, const T* src, int num_dimensions) {
  static_assert(sizeof(VecType) % sizeof(T) == 0,
                "Unsopported combination of copy type and vector type");
  constexpr int vec_width = sizeof(VecType) / sizeof(T);

  int start_id =
      ((sizeof(VecType) - (intptr_t)src % sizeof(VecType)) % sizeof(VecType)) / sizeof(T);
  int end_id = num_dimensions -
               (((intptr_t)src + (num_dimensions * sizeof(T))) % sizeof(VecType)) / sizeof(T);

  for (int i = threadIdx.x; i < num_dimensions; i += blockDim.x) {
    if (i < start_id || i >= end_id) {
      shmem_dst[i] = src[i];
    }
  }

  for (int i = start_id + threadIdx.x * vec_width; i < end_id; i += blockDim.x * vec_width) {
    auto cur_vec_src = reinterpret_cast<const VecType*>(src + i);
    auto val = *cur_vec_src;
    vec2arr(shmem_dst + i, val);
  }
}

template <typename VecType, typename T>
__device__ inline void shmem2global_vec(T* dst, const T* shmem_src, int num_dimensions) {
  static_assert(sizeof(VecType) % sizeof(T) == 0,
                "Unsopported combination of copy type and vector type");
  constexpr int vec_width = sizeof(VecType) / sizeof(T);

  int start_id =
      ((sizeof(VecType) - (intptr_t)dst % sizeof(VecType)) % sizeof(VecType)) / sizeof(T);
  int end_id = num_dimensions -
               (((intptr_t)dst + (num_dimensions * sizeof(T))) % sizeof(VecType)) / sizeof(T);

  for (int i = threadIdx.x; i < num_dimensions; i += blockDim.x) {
    if (i < start_id || i >= end_id) {
      dst[i] = shmem_src[i];
    }
  }

  for (int i = start_id + threadIdx.x * vec_width; i < end_id; i += blockDim.x * vec_width) {
    auto cur_vec_dst = reinterpret_cast<VecType*>(dst + i);
    *cur_vec_dst = arr2vec<VecType>(shmem_src + i);
  }
}

template <typename SrcT, typename DstT>
__device__ inline void convert(DstT* dst, const SrcT* src, int num_dimensions) {
  for (int i = threadIdx.x; i < num_dimensions; i += blockDim.x) {
    dst[i] = (DstT)src[i];
  }
}

template <typename Config, typename CopyDesc>
__global__ void aligned(CopyDesc copy_desc) {
  // Grid  ( arbitrary,           1,         1 )
  // Block ( threads_per_element, arbitrary, 1 )

  using VecType = typename Config::VecType;
  using SrcT = typename CopyDesc::SrcT;
  using DstT = typename CopyDesc::DstT;
  constexpr int ndests = CopyDesc::ndests;

  // static_assert here requires C++17 constexpr if (or ugly WARs)
  //   static_assert(Config::Aligned, "Aligned kernel needs Config::Aligned == true");
  //   static_assert(std::is_same<SrcT, DstT>::value,
  //     "Aligned kernel only supports same src and dst types");

  assert(Config::Aligned);
  assert((std::is_same<SrcT, DstT>::value));

  constexpr int vec_width = sizeof(VecType) / sizeof(SrcT);

  const int src_start_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_dimensions = copy_desc.num_dimensions();

  for (int src_id = src_start_id; src_id < copy_desc.src_buf_size();
       src_id += gridDim.x * blockDim.y) {
    auto copy_details = copy_desc.get_details(src_id);

    assert((intptr_t)copy_details.src_ptr % sizeof(VecType) == 0);
    auto src_ptr_vec = reinterpret_cast<const VecType*>(copy_details.src_ptr);

    for (int i = threadIdx.x; i < num_dimensions / vec_width; i += blockDim.x) {
      VecType val = src_ptr_vec[i];
#pragma unroll
      for (int dst_id = 0; dst_id < ndests; dst_id++) {
        if (copy_details.do_copy[dst_id]) {
          auto dst_ptr_vec = reinterpret_cast<VecType*>(copy_details.dst_ptr[dst_id]);
          assert((intptr_t)dst_ptr_vec % sizeof(VecType) == 0);
          dst_ptr_vec[i] = val;
        }
      }
    }
  }
}

template <typename Config, typename CopyDesc>
__global__ void arbitrary(CopyDesc copy_desc) {
  // Grid  ( arbitrary,           1,         1 )
  // Block ( threads_per_element, arbitrary, 1 )

  using VecType = typename Config::VecType;
  using SrcT = typename CopyDesc::SrcT;
  using DstT = typename CopyDesc::DstT;
  constexpr int ndests = CopyDesc::ndests;

  static_assert(sizeof(VecType) % sizeof(SrcT) == 0,
                "Unsopported combination of copy type and vector type");
  static_assert(sizeof(VecType) % sizeof(DstT) == 0,
                "Unsopported combination of copy type and vector type");

  const int num_elems = copy_desc.src_buf_size();
  const int src_start_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_dimensions = copy_desc.num_dimensions();

  extern __shared__ char shmem[];
  SrcT* src_scratch = reinterpret_cast<SrcT*>(shmem);
  DstT* dst_scratch = reinterpret_cast<DstT*>(src_scratch + num_dimensions * blockDim.y);

  constexpr bool use_warp_sync = Config::block.x == 32;
  auto sync = [use_warp_sync]() {
    if (use_warp_sync) {
      __syncwarp();
    } else {
      __syncthreads();
    }
  };

  for (int src_id = src_start_id; src_id - threadIdx.y < num_elems;
       src_id += gridDim.x * blockDim.y) {
    typename std::result_of<decltype (&CopyDesc::get_details)(CopyDesc, size_t)>::type copy_details;

    SrcT* my_src_scratch = src_scratch + num_dimensions * threadIdx.y;
    DstT* my_dst_scratch = dst_scratch + num_dimensions * threadIdx.y;
    if (src_id < num_elems) {
      copy_details = copy_desc.get_details(src_id);
      global2shmem_vec<VecType>(my_src_scratch, copy_details.src_ptr, num_dimensions);
    }
    sync();
    convert(my_dst_scratch, my_src_scratch, num_dimensions);
    sync();
    if (src_id < num_elems) {
#pragma unroll
      for (int dst_id = 0; dst_id < ndests; dst_id++) {
        if (copy_details.do_copy[dst_id]) {
          shmem2global_vec<VecType>(copy_details.dst_ptr[dst_id], my_dst_scratch, num_dimensions);
        }
      }
    }
    sync();
  }
}

}  // namespace ShuffleKernels

namespace HugeCTR {

// Workaround for if constexpr in C++14
template <typename Config, typename CopyDesc>
void __shuffle_dispatch_aligned(CopyDesc copy_desc, size_t expected_elements, cudaStream_t stream) {
  dim3 block = Config::block;
  size_t grid = std::max((size_t)1, std::min((size_t)100000, expected_elements / block.y / 2 + 1));

  ShuffleKernels::aligned<Config><<<(uint32_t)grid, block, 0, stream>>>(copy_desc);
}

template <typename Config, typename CopyDesc>
void __shuffle_dispatch(CopyDesc copy_desc, size_t expected_elements, cudaStream_t stream) {
  dim3 block = Config::block;
  size_t grid = std::max((size_t)1, std::min((size_t)100000, expected_elements / block.y / 2 + 1));

  size_t shmem_size = copy_desc.num_dimensions() * block.y *
                      (sizeof(typename CopyDesc::SrcT) + sizeof(typename CopyDesc::DstT));

  ShuffleKernels::arbitrary<Config><<<grid, block, shmem_size, stream>>>(copy_desc);
}

template <typename Config = ShuffleConfigs::DefaultAligned, typename CopyDesc>
void shuffle(CopyDesc copy_desc, cudaStream_t stream, size_t expected_elements = 10000) {
  assert((sizeof(typename CopyDesc::SrcT) * copy_desc.num_dimensions()) % sizeof(int) == 0 &&
         (sizeof(typename CopyDesc::DstT) * copy_desc.num_dimensions()) % sizeof(int) == 0);

  if (std::is_same<typename CopyDesc::SrcT, typename CopyDesc::DstT>::value && Config::Aligned &&
      (copy_desc.num_dimensions() % sizeof(typename Config::VecType) == 0)) {
    __shuffle_dispatch_aligned<Config, CopyDesc>(copy_desc, expected_elements, stream);
  } else {
    __shuffle_dispatch<Config, CopyDesc>(copy_desc, expected_elements, stream);
  }
}

}  // namespace HugeCTR

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

#pragma once

/*
 *
 * CopyInfo struct needs to provide:
 *  T : as the type of the buffers
 *  __host__ __device__ int num_dimensions() : number of elements of type T to copy in one
 * transaction
 *  __device__ size_t src_buf_size() : size of the src buffer
 *  __device__ T* get_src_ptr(src_id) : pointer to the num_dimensions() values of type T that will
 * be copied
 *  __device__ int num_dests(src_id) : number of the destinations for the src_id index
 *  __device__ T* get_dst_ptr(src_id, dst_id) : destination pointer (same requirements as src
 * pointer)
 *
 */

namespace HugeCTR {
namespace CopyDescriptors {

template <typename SrcType, typename DstType, int NumDests>
struct CopyDetails {
  const SrcType* src_ptr;
  DstType* dst_ptr[NumDests];
  bool do_copy[NumDests];
};

template <typename SrcType, typename DstType, int NumDests,
          typename LambdaNumElemType, typename LambdaDetailsType>
struct OneToOne {
  using SrcT = SrcType;
  using DstT = DstType;
  static constexpr int ndests = NumDests;

  __host__ __device__ __forceinline__ int num_dimensions() { return num_dimensions_; }
  __device__ __forceinline__ size_t src_buf_size() { return get_num_elems_(); }
  __device__ __forceinline__ CopyDetails<SrcT, DstT, NumDests> get_details(size_t id) {
    return get_details_(id);
  }

  uint32_t num_dimensions_;
  LambdaNumElemType get_num_elems_;
  LambdaDetailsType get_details_;
};

template <typename SrcType, typename DstType, int NumDests,
          typename LambdaNumElemType, typename LambdaDetailsType>
OneToOne<SrcType, DstType, NumDests, LambdaNumElemType, LambdaDetailsType> make_OneToOne(
    uint32_t num_dimensions, LambdaNumElemType get_num_elems, LambdaDetailsType get_details) {
  return {num_dimensions, get_num_elems, get_details};
}

}  // namespace CopyDescriptors
}  // namespace HugeCTR
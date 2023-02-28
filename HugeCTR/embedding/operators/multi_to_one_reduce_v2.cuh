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

#include <stdio.h>

#include <embedding/operators/generic_lookup.cuh>

#define EV_NUM 32
#define BLOCK_SIZE_NEW 64
#define WARP_SIZE 32
#define MAX_BLOKC_SIZE_PER_SM 2048

namespace embedding {

template <typename CopyDesc, typename DST_T, int kMaxElemPerThread, int kWarpSize>
__global__ void multi_to_one_reduce_vec4_v2(CopyDesc copy_desc, DST_T* partial_buffer,
                                            uint32_t* partial_dst_ids, int32_t* partial_ev_length,
                                            int max_ev_length) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;

  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warp_num = blockDim.x >> 5;
  int local_sample_num = EV_NUM;
  constexpr int copy_width = 4;
  int global_index = EV_NUM * (blockIdx.x * warp_num + warp_id);
  {
    if (global_index >= copy_desc.num_vec()) return;
    local_sample_num = local_sample_num < copy_desc.num_vec() - global_index
                           ? local_sample_num
                           : copy_desc.num_vec() - global_index;
  }

  Vec4T<float> accum[kMaxElemPerThread];
  uint32_t tmp_dst_id;
  int vec_length = -1;
  for (int sp = 0; sp < local_sample_num; ++sp) {
    {
      tmp_dst_id = copy_desc.get_dst_id(global_index);
      const src_type* tmp_src = copy_desc.get_src_ptr(global_index);
      vec_length = copy_desc.get_src_vec_length(global_index);
      for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
        Vec4T<src_type> src_elem;
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min(vec_length - idx4, copy_width);
        src_elem.load(tmp_src + idx4, n);
        accum[i].accumulate(src_elem);
      }
    }

    // when key is change , write to dst
    if (sp < local_sample_num - 1) {
      uint32_t new_id = copy_desc.get_dst_id(global_index + 1);
      if (new_id != tmp_dst_id) {
        dst_type* tmp_dst = copy_desc.get_dst_ptr(global_index);
        for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length;
             ++i) {
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(vec_length - idx4, copy_width);
          accum[i].store(tmp_dst + idx4, n);
          accum[i].reset();
        }
      }
    }
    global_index++;
  }

  if (vec_length != -1) {
    bool is_last = true;
    dst_type* tmp_dst;
    if (global_index < copy_desc.num_vec()) {
      auto next_id = copy_desc.get_dst_id(global_index);
      if (tmp_dst_id == next_id) is_last = false;
    }

    if (is_last) {
      tmp_dst = copy_desc.get_dst_ptr(global_index - 1);
      for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min(vec_length - idx4, copy_width);
        accum[i].store(tmp_dst + idx4, n);
        accum[i].reset();
      }
      if (lane_id == 0) {
        partial_ev_length[blockIdx.x * warp_num + warp_id] = -1;
      }

    } else {
      tmp_dst = partial_buffer + (blockIdx.x * warp_num + warp_id) * max_ev_length;
      for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min(vec_length - idx4, copy_width);
        accum[i].store(tmp_dst + idx4, n);
        accum[i].reset();
      }

      if (lane_id == 0) {
        partial_ev_length[blockIdx.x * warp_num + warp_id] = vec_length;
        partial_dst_ids[blockIdx.x * warp_num + warp_id] = tmp_dst_id;
      }
    }
  }

  return;
}

template <typename CopyDesc, int kMaxElemPerThread, int kWarpSize>
__global__ void multi_to_one_reduce_final_v2(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;

  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warp_num = blockDim.x >> 5;
  int local_sample_num = EV_NUM;
  constexpr int copy_width = 4;
  int global_index = EV_NUM * (blockIdx.x * warp_num + warp_id);
  {
    if (global_index >= copy_desc.num_vec()) return;
    local_sample_num = local_sample_num < copy_desc.num_vec() - global_index
                           ? local_sample_num
                           : copy_desc.num_vec() - global_index;
  }

  Vec4T<float> accum[kMaxElemPerThread];
  uint32_t tmp_dst_id;
  int vec_length = -1;
  for (int sp = 0; sp < local_sample_num; ++sp) {
    vec_length = copy_desc.get_src_vec_length(global_index);
    if (vec_length != -1) {
      tmp_dst_id = copy_desc.get_dst_id(global_index);
      const src_type* tmp_src = copy_desc.get_src_ptr(global_index);
      for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
        Vec4T<src_type> src_elem;
        int idx4 = 4 * kWarpSize * i + 4 * lane_id;
        int n = min(vec_length - idx4, copy_width);
        src_elem.load(tmp_src + idx4, n);
        accum[i].accumulate(src_elem);
      }

      // when key is change , write to dst
      if (sp < local_sample_num - 1) {
        uint32_t new_id = copy_desc.get_dst_id(global_index + 1);
        int new_vec_length = copy_desc.get_src_vec_length(global_index + 1);
        if (new_id != tmp_dst_id || vec_length != new_vec_length) {
          dst_type* tmp_dst = copy_desc.get_dst_ptr(global_index);
          for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length;
               ++i) {
            int idx4 = 4 * kWarpSize * i + 4 * lane_id;
            int n = min(vec_length - idx4, copy_width);
            accum[i].atomic_store_accum(tmp_dst + idx4, n);
            accum[i].reset();
          }
        }
      }
    }
    global_index++;
  }

  if (vec_length != -1) {
    dst_type* tmp_dst = copy_desc.get_dst_ptr(global_index - 1);
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].atomic_store_accum(tmp_dst + idx4, n);
    }
  }

  return;
}

template <typename CopyDesc1, typename CopyDesc2, typename DST_T, int kWarpSize = 32>
void multi_to_one_reduce_v2(CopyDesc1 copy_desc1, CopyDesc2 copy_desc2, DST_T* partial_buffer,
                            uint32_t* partial_dst_ids, int32_t* partial_ev_length,
                            size_t max_input_num, int max_ev_length, cudaStream_t stream) {
  int grid_size = (max_input_num - 1) / BLOCK_SIZE_NEW + 1;
  int block_size = BLOCK_SIZE_NEW;
  if (max_ev_length <= 128) {
    if (grid_size > 1) {
      multi_to_one_reduce_vec4_v2<CopyDesc1, DST_T, 1, kWarpSize>
          <<<grid_size, block_size, 0, stream>>>(copy_desc1, partial_buffer, partial_dst_ids,
                                                 partial_ev_length, max_ev_length);
      int final_grid_size = (grid_size - 1) / EV_NUM + 1;
      multi_to_one_reduce_final_v2<CopyDesc2, 1, kWarpSize>
          <<<final_grid_size, block_size, 0, stream>>>(copy_desc2);
    } else {
      multi_to_one_reduce_final_v2<CopyDesc1, 1, kWarpSize>
          <<<1, block_size, 0, stream>>>(copy_desc1);
    }

  } else if (max_ev_length <= 256) {
    if (grid_size > 1) {
      multi_to_one_reduce_vec4_v2<CopyDesc1, DST_T, 1, kWarpSize>
          <<<grid_size, block_size, 0, stream>>>(copy_desc1, partial_buffer, partial_dst_ids,
                                                 partial_ev_length, max_ev_length);

      int final_grid_size = (grid_size - 1) / EV_NUM + 1;
      multi_to_one_reduce_final_v2<CopyDesc2, 2, kWarpSize>
          <<<final_grid_size, block_size, 0, stream>>>(copy_desc2);
    } else {
      multi_to_one_reduce_final_v2<CopyDesc1, 2, kWarpSize>
          <<<1, block_size, 0, stream>>>(copy_desc1);
    }

  } else if (max_ev_length <= 1024) {
    // multi_to_one_reduce_large_ev<SRC_TENSOR, DST_TENSOR, DST_T, kWarpSize>
    //    <<<grid_size, block_size, 0, stream>>>(src, dst, key_array, ind_array, dst_idx_array,
    //                                           dst_offset_array, partial_buffer,
    //                                           partial_key_buffer, partial_ev_length,
    //                                           partial_dst_offset_array, embedding_vector_num,
    //                                           max_ev_length);
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "HugeCTR does not support emb vector size > 256");
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "HugeCTR does not support emb vector size >= 1024");
  }
}

template <typename CopyDesc1>
void multi_to_one_reduce_v2(CopyDesc1 multi_to_one_desc_first_stage,
                            const ReductionIndices& reduction_indices,
                            const KernelParams& kernel_params,
                            PartialReduceResult& partial_reduce_result, Wgrad& wgrad,
                            int max_ev_size, cudaStream_t stream) {
  auto partial_grad_ev_ptr = partial_reduce_result.partial_wgrad_new.data<float>();
  auto partial_ev_length_ptr = partial_reduce_result.partial_ev_length_new.data<int32_t>();
  auto partial_dst_id_array_ptr = partial_reduce_result.partial_dst_id_array_new.data<uint32_t>();

  const int* table_ids_ptr = wgrad.table_ids.data<int>();
  const int* table_id_to_ev_size_ptr = wgrad.attr.table_id_to_ev_size.data<int>();
  const uint32_t* dst_ev_start_indices_ptr = wgrad.ev_start_indices.data<uint32_t>();
  float* dst_ptr = wgrad.data.data<float>();
  size_t second_num = (reduction_indices.num_elements - 1) / EV_NUM + 1;
  auto multi_to_one_desc_second_stage = make_MultiToOne_reduce_new<float, float>(
      [=] __device__() { return second_num; },
      [=] __device__(int i) { return partial_ev_length_ptr[i]; },
      [=] __device__(int i) { return partial_dst_id_array_ptr[i]; },
      [=] __device__(int i) { return partial_grad_ev_ptr + i * max_ev_size; },

      [=] __device__(int i) {
        auto tmp_index = partial_dst_id_array_ptr[i];
        return dst_ptr + dst_ev_start_indices_ptr[tmp_index];
      });

  multi_to_one_reduce_v2(multi_to_one_desc_first_stage, multi_to_one_desc_second_stage,
                         partial_grad_ev_ptr, partial_dst_id_array_ptr, partial_ev_length_ptr,
                         partial_reduce_result.max_input_num, max_ev_size, stream);
}

}  // namespace embedding

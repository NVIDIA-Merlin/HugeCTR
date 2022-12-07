#pragma once

#include <stdio.h>

#include "generic_lookup.cuh"

#define MIN_EV_NUM 10
#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define MAX_BLOKC_SIZE_PER_SM 2048

namespace embedding {

__device__ __forceinline__ void find_destination_binary(uint32_t target_key, int l, int r,
                                                        uint32_t* key_buffer, int* dest_index) {
  if (key_buffer[l] != target_key) {
    return;
  }
  while (l <= r) {
    int mid = (l + r) >> 1;
    if (key_buffer[mid] == target_key) {
      l = mid + 1;
    } else if (key_buffer[mid] != target_key) {
      r = mid - 1;
    }
  }
  *dest_index = r;
  return;
}

template <typename CopyDesc, typename DST_T, int kMaxElemPerThread, int kWarpSize>
__global__ void multi_to_one_reduce_vec4(CopyDesc copy_desc, DST_T* partial_buffer,
                                         uint32_t* partial_key_buffer, int* partial_ev_length,
                                         uint32_t* partial_dst_idx_array, int max_ev_length) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  int lane_id = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warp_num = blockDim.x >> 5;
  constexpr int copy_width = 4;

  int local_sample_num = (copy_desc.num_vec_) / gridDim.x / warp_num;
  size_t start_offset;
  // calculate a warp in charge of how many embedding vectors
  {
    int local_reside = (copy_desc.num_vec_) % (gridDim.x * warp_num);
    if (blockIdx.x * warp_num + warp_id < local_reside) {
      local_sample_num += 1;
      start_offset = (blockIdx.x * warp_num + warp_id) * local_sample_num;
    } else {
      start_offset = local_reside * (local_sample_num + 1) +
                     (blockIdx.x * warp_num + warp_id - local_reside) * local_sample_num;
    }
    if (start_offset + local_sample_num > copy_desc.num_vec_)
      local_sample_num = copy_desc.num_vec_ - start_offset;
  }

  extern __shared__ DST_T smem_buffer[];
  DST_T* ev_buffer = smem_buffer;
  uint32_t* last_key = (uint32_t*)(&smem_buffer[max_ev_length * warp_num]);
  int* dest_index = (int*)(&last_key[warp_num]);
  int* smem_ev_length = &dest_index[warp_num];

  Vec4T<float> accum[kMaxElemPerThread];
  uint32_t tmp_key;
  vec_length_type vec_length = 0;
  for (int sp = 0; sp < local_sample_num; ++sp) {
    tmp_key = copy_desc.get_key(sp + start_offset);
    const src_type* tmp_src = copy_desc.get_src_ptr(sp + start_offset);
    vec_length = copy_desc.get_src_vec_length_(sp + start_offset);
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      Vec4T<src_type> src_elem;
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      src_elem.load(tmp_src + idx4, n);
      accum[i].accumulate(src_elem);
    }

    // when key is change , write to dst
    if (sp < local_sample_num - 1) {
      uint32_t new_key = copy_desc.get_key(sp + start_offset + 1);
      if (new_key != tmp_key) {
        dst_type* tmp_dst = copy_desc.get_dst_ptr(sp + start_offset);
        for (int i = 0;
             i < kMaxElemPerThread &&
             4 * kWarpSize * i + 4 * lane_id < copy_desc.get_dst_vec_length(sp + start_offset);
             ++i) {
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(vec_length - idx4, copy_width);
          accum[i].store(tmp_dst + idx4, n);
          accum[i].reset();
        }
      }
    }
  }

  // write to final embedding vector to shared memory , because always keep a embedding vector
  // remain.
  for (int i = 0; i < kMaxElemPerThread; ++i) {
    int ev_offset = ((i * kWarpSize + lane_id) << 2);
    if (ev_offset < vec_length) {
      int smem_offset = warp_id * max_ev_length + ev_offset;
      int n = min(vec_length - ev_offset, copy_width);
      accum[i].store(ev_buffer + smem_offset, n);
    }
  }

  // write remain key and embedding vector length.
  if (lane_id == 0) {
    last_key[warp_id] = tmp_key;
    smem_ev_length[warp_id] = vec_length;
  }
  __syncthreads();

  // find same remain key between wraps
  if (lane_id == 0) {
    int tmp_dest_index = warp_id;
    if (warp_id != warp_num - 1) {
      find_destination_binary(tmp_key, warp_id + 1, warp_num - 1, last_key, &tmp_dest_index);
    }
    dest_index[warp_id] = tmp_dest_index;
  }

  __syncthreads();

  // atomic reduce same remain key between wraps
  if (dest_index[warp_id] != warp_id) {
    int local_warp_offset = warp_id * max_ev_length;
    int dest_warp_offset = dest_index[warp_id] * max_ev_length;

#pragma unroll
    for (int i = lane_id; i < smem_ev_length[warp_id]; i += kWarpSize) {
      atomicAdd(ev_buffer + dest_warp_offset + i, ev_buffer[local_warp_offset + i]);
    }
  }

  __syncthreads();
  // if a warp remain key not same with the last warp of this block , atomic add to dst
  if (dest_index[warp_id] == warp_id && warp_id != warp_num - 1) {
    dst_type* tmp_dst = copy_desc.get_dst_ptr(local_sample_num - 1 + start_offset);
    int tmp_ev_length = smem_ev_length[warp_id];
#pragma unroll
    for (int i = lane_id; i < tmp_ev_length; i += kWarpSize) {
      atomicAdd(tmp_dst + i, ev_buffer[warp_id * max_ev_length + i]);
    }
  } else if (warp_id == warp_num - 1) {
    // write the last warp of this block embedding vector into partial buffer
    int block_id = blockIdx.x;
    partial_buffer = partial_buffer + block_id * max_ev_length;
    int tmp_ev_length = smem_ev_length[warp_id];
#pragma unroll
    for (int i = lane_id; i < tmp_ev_length; i += kWarpSize) {
      partial_buffer[i] = ev_buffer[warp_id * max_ev_length + i];
    }

    if (lane_id == 0) {
      partial_key_buffer[block_id] = last_key[warp_id];
      partial_ev_length[block_id] = smem_ev_length[warp_id];
      partial_dst_idx_array[block_id] =
          copy_desc.get_dst_unique_id(start_offset + local_sample_num - 1);
    }
  }

  return;
}

template <typename CopyDesc, int kMaxElemPerThread, int kWarpSize>
__launch_bounds__(1024, 1) __global__
    void multi_to_one_reduce_final(CopyDesc copy_desc, int max_ev_length) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  int lane_id = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warp_num = blockDim.x >> 5;

  constexpr int copy_width = 4;

  int local_sample_num = copy_desc.num_vec_ / warp_num;
  size_t start_offset;

  {
    int local_reside = copy_desc.num_vec_ % warp_num;
    if (warp_id < local_reside) {
      local_sample_num += 1;
      start_offset = warp_id * local_sample_num;
    } else {
      start_offset =
          local_reside * (local_sample_num + 1) + (warp_id - local_reside) * local_sample_num;
    }
    if (start_offset + local_sample_num > copy_desc.num_vec_)
      local_sample_num = copy_desc.num_vec_ - start_offset;
  }

  extern __shared__ dst_type smem_buffer[];
  dst_type* ev_buffer = smem_buffer;
  uint32_t* last_key = (uint32_t*)(&smem_buffer[max_ev_length * warp_num]);
  int* dest_index = (int*)(&last_key[warp_num]);
  int* smem_ev_length = &dest_index[warp_num];

  Vec4T<float> accum[kMaxElemPerThread];
  uint32_t tmp_key;
  vec_length_type vec_length = 0;

  for (int sp = 0; sp < local_sample_num; ++sp) {
    tmp_key = copy_desc.get_key(sp + start_offset);
    const src_type* tmp_src = copy_desc.get_src_ptr(sp + start_offset);
    vec_length = copy_desc.get_src_vec_length_(sp + start_offset);
    for (int i = 0; i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length; ++i) {
      Vec4T<src_type> src_elem;
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      src_elem.load(tmp_src + idx4, n);
      accum[i].accumulate(src_elem);
    }

    if (sp < local_sample_num - 1) {
      uint32_t new_key = copy_desc.get_key(sp + start_offset + 1);
      if (new_key != tmp_key) {
        dst_type* tmp_dst = copy_desc.get_dst_ptr(sp + start_offset);
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

  for (int i = 0; i < kMaxElemPerThread; ++i) {
    int ev_offset = ((i * kWarpSize + lane_id) << 2);
    if (ev_offset < vec_length) {
      int smem_offset = warp_id * max_ev_length + ev_offset;
      int n = min(vec_length - ev_offset, copy_width);
      accum[i].store(ev_buffer + smem_offset, n);
    }
  }

  if (lane_id == 0) {
    last_key[warp_id] = tmp_key;
    smem_ev_length[warp_id] = vec_length;
  }

  __syncthreads();

  if (lane_id == 0) {
    int tmp_dest_index = warp_id;
    if (warp_id != warp_num - 1)
      find_destination_binary(tmp_key, warp_id + 1, warp_num - 1, last_key, &tmp_dest_index);
    dest_index[warp_id] = tmp_dest_index;
  }
  __syncthreads();

  if (dest_index[warp_id] != warp_id) {
    int local_warp_offset = warp_id * max_ev_length;
    int dest_warp_offset = dest_index[warp_id] * max_ev_length;
#pragma unroll
    for (int i = lane_id; i < smem_ev_length[warp_id]; i += kWarpSize) {
      atomicAdd(ev_buffer + dest_warp_offset + i, ev_buffer[local_warp_offset + i]);
    }
  }

  __syncthreads();
  if (dest_index[warp_id] == warp_id) {
    dst_type* tmp_dst = copy_desc.get_dst_ptr(local_sample_num - 1 + start_offset);
    int tmp_ev_length = smem_ev_length[warp_id];
#pragma unroll
    for (int i = lane_id; i < tmp_ev_length; i += kWarpSize) {
      atomicAdd(tmp_dst + i, ev_buffer[warp_id * max_ev_length + i]);
    }
    return;
  }

  return;
}

void get_kernel_config(int device_sms, int* grid_size, int* block_size, int max_ev_length,
                       size_t embedding_vector_num) {
  int tmp_grid_size;
  int tmp_block_size = BLOCK_SIZE;
  if (max_ev_length <= 256) {
    tmp_grid_size = int((embedding_vector_num) / (BLOCK_SIZE / WARP_SIZE));

    if (tmp_grid_size > device_sms * (MAX_BLOKC_SIZE_PER_SM / BLOCK_SIZE)) {
      tmp_grid_size = device_sms * (MAX_BLOKC_SIZE_PER_SM / BLOCK_SIZE);
    } else if (tmp_grid_size == 0) {
      tmp_grid_size = 1;
      tmp_block_size = embedding_vector_num * WARP_SIZE;
    }
    *grid_size = tmp_grid_size;
    *block_size = tmp_block_size;

  } else {
    int num_block_per_sm = MAX_BLOKC_SIZE_PER_SM / max_ev_length;
    tmp_grid_size = num_block_per_sm * device_sms;
    tmp_block_size = max_ev_length;
    if (embedding_vector_num < (size_t)num_block_per_sm) {
      tmp_grid_size = embedding_vector_num;
    }
    *block_size = tmp_block_size;
    *grid_size = tmp_grid_size;
  }
}

template <typename CopyDesc1, typename CopyDesc2, typename DST_T, int kWarpSize = 32>
void multi_to_one_reduce(CopyDesc1 copy_desc1, CopyDesc2 copy_desc2, DST_T* partial_buffer,
                         uint32_t* partial_key_buffer, int* partial_ev_length,
                         uint32_t* partial_dst_offset_array, int device_sms, int max_ev_length,
                         cudaStream_t stream) {
  int grid_size;
  int block_size;
  get_kernel_config(device_sms, &grid_size, &block_size, max_ev_length, copy_desc1.num_vec_);
  if (max_ev_length <= 128) {
    if (grid_size > 1) {
      multi_to_one_reduce_vec4<CopyDesc1, DST_T, 1, kWarpSize>
          <<<grid_size, block_size,
             max_ev_length*(block_size / WARP_SIZE) * sizeof(DST_T) +
                 3 * sizeof(int) * (block_size / WARP_SIZE),
             stream>>>(copy_desc1, partial_buffer, partial_key_buffer, partial_ev_length,
                       partial_dst_offset_array, max_ev_length);
      int final_block_size = grid_size < 32 ? grid_size * WARP_SIZE : 1024;
      copy_desc2.num_vec_ = grid_size;
      multi_to_one_reduce_final<CopyDesc2, 1, kWarpSize>
          <<<1, final_block_size,
             max_ev_length * final_block_size / WARP_SIZE * sizeof(DST_T) +
                 3 * sizeof(int) * final_block_size / WARP_SIZE,
             stream>>>(copy_desc2, max_ev_length);
    } else {
      multi_to_one_reduce_final<CopyDesc1, 1, kWarpSize>
          <<<1, block_size,
             max_ev_length * block_size / WARP_SIZE * sizeof(DST_T) +
                 3 * sizeof(int) * block_size / WARP_SIZE,
             stream>>>(copy_desc1, max_ev_length);
    }

  } else if (max_ev_length <= 256) {
    if (grid_size > 1) {
      multi_to_one_reduce_vec4<CopyDesc1, DST_T, 2, kWarpSize>
          <<<grid_size, block_size,
             max_ev_length*(block_size / WARP_SIZE) * sizeof(DST_T) +
                 3 * sizeof(int) * (block_size / WARP_SIZE),
             stream>>>(copy_desc1, partial_buffer, partial_key_buffer, partial_ev_length,
                       partial_dst_offset_array, max_ev_length);
      int final_block_size = grid_size < 32 ? grid_size * WARP_SIZE : 1024;
      copy_desc2.num_vec_ = grid_size;
      multi_to_one_reduce_final<CopyDesc2, 2, kWarpSize>
          <<<1, final_block_size,
             max_ev_length * final_block_size / WARP_SIZE * sizeof(DST_T) +
                 3 * sizeof(int) * final_block_size / WARP_SIZE,
             stream>>>(copy_desc2, max_ev_length);
    } else {
      multi_to_one_reduce_final<CopyDesc1, 2, kWarpSize>
          <<<1, block_size,
             max_ev_length * block_size / WARP_SIZE * sizeof(DST_T) +
                 3 * sizeof(int) * block_size / WARP_SIZE,
             stream>>>(copy_desc1, max_ev_length);
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

}  // namespace embedding

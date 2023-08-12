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

#include <hps/quantize.hpp>

namespace HugeCTR {

#define FINAL_MASK 0xffffffff

template <typename InT, typename OutT>
Quantize<InT, OutT>::Quantize(const bool shift_to_uint8, const bool round_before_cast)
    : _shift_to_uint8(shift_to_uint8), _round_before_cast(round_before_cast) {}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

template <typename InT, typename OutT>
__global__ void quantize_kernel(const InT* input, size_t emb_vec_size, float* scales, OutT* output,
                                bool round_before_cast) {
  constexpr float min_scaling_factor = 1 / (FP8_E4M3_MAX * CLAMP);
  int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  float amax_val = 0.0f;
  for (int k_i = tidx; k_i < emb_vec_size; k_i += blockDim.x) {
    float val = fabs(static_cast<InT>(input[k_i + bidx * emb_vec_size]));
    if (amax_val < val) {
      amax_val = val;
    }
  }
  const float block_amax_val = blockReduceMax(amax_val);
  __shared__ float scale;
  if (tidx == 0) {
    if constexpr (is_fp8<OutT>::value) {
      if (round_before_cast) {
        scale = (float)std::max(block_amax_val / FP8_E4M3_MAX, min_scaling_factor);
        float exp = ceilf(log2f(scale));
        scale = powf(2.0, exp);
        scales[bidx] = exp < 0 ? scale : 1 / scale;
      } else {
        scale = (float)std::max(block_amax_val / FP8_E4M3_MAX, min_scaling_factor);
        scales[bidx] = scale;
      }
    } else {
      scale = block_amax_val != 0.f ? (float)(INT8_MAX / block_amax_val) : 1.f;
      scales[bidx] = scale;
    }
  }
  __syncthreads();
  input += blockIdx.x * emb_vec_size;
  output += blockIdx.x * emb_vec_size;
  for (; tidx < emb_vec_size; tidx += blockDim.x) {
    if constexpr (is_fp8<OutT>::value)
      output[tidx] = OutT(float(input[tidx]) / float(scales[bidx]));
    else
      output[tidx] = OutT(input[tidx] * scales[bidx]);
  }
}

template <typename InT, typename OutT>
void Quantize<InT, OutT>::quantize(InT* input, OutT* output, InT* scale, size_t batch_size,
                                   size_t emb_vec_size, cudaStream_t stream = 0) const {
  dim3 grid(batch_size);
  dim3 block((emb_vec_size + 31) / 32 * 32);
  if (block.x > 1024) {
    block.x = 1024;
  }
  quantize_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<InT*>(input), emb_vec_size,
                                              reinterpret_cast<float*>(scale),
                                              reinterpret_cast<OutT*>(output), _round_before_cast);
  cudaStreamSynchronize(stream);
}

template class Quantize<float, int8_t>;
template class Quantize<float, __nv_fp8_e4m3>;
}  // namespace HugeCTR
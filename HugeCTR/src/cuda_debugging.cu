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

#include <cuda_debugging.cuh>

namespace HugeCTR {

namespace {

__device__ char* kernel_strcpy(const char* dst_base, char* dst_cur, const char* src) {
    int n = 0;
    while(*src != '\0') {
        if ((size_t)(dst_cur - dst_base) >= KERNEL_PRINTF_MAX_LENGTH ) { return dst_cur; }
        *dst_cur++ = *src++;
        n++;
    }
    return dst_cur;
}

__device__ char* kernel_utoa(const char* dst_base, char* dst_cur, unsigned val) {
    int n = 0;
    do {
        if ((size_t)(dst_cur - dst_base) >= KERNEL_PRINTF_MAX_LENGTH ) { return dst_cur + n; }
        unsigned dgt = val % 10;
        char ch = '0' + dgt;
        *(dst_cur + n) = ch;
        n++;
    } while (val /= 10);
    dst_cur[n] = '\0';

    for (int i = 0; i < n / 2; i++) {
        char tmp = dst_cur[n - i - 1];
        dst_cur[n - i - 1] = dst_cur[i];
        dst_cur[i] = tmp;
    }

    return dst_cur + n;
}

__device__ char* kernel_printf_info(char* dst){
  unsigned grid_id;
  asm("mov.u32 %0, %gridid;" : "=r"(grid_id) : );
  int device_id;
  cudaGetDevice(&device_id);
  unsigned sm_id;
  asm("mov.u32 %0, %smid;" : "=r"(sm_id) : );
  unsigned warp_id;
  asm("mov.u32 %0, %warpid;" : "=r"(warp_id) : );
  unsigned lane_id;
  asm("mov.u32 %0, %laneid;" : "=r"(lane_id) : );

  char* nxt = dst;
  nxt = kernel_strcpy(dst, nxt, "[HUGECTR][KERNEL][device ");
  nxt = kernel_utoa(dst, nxt, device_id);
  nxt = kernel_strcpy(dst, nxt, ", grid ");
  nxt = kernel_utoa(dst, nxt, grid_id);
  nxt = kernel_strcpy(dst, nxt, ", block (");
  nxt = kernel_utoa(dst, nxt, blockIdx.x);
  nxt = kernel_strcpy(dst, nxt, ",");
  nxt = kernel_utoa(dst, nxt, blockIdx.y);
  nxt = kernel_strcpy(dst, nxt, ",");
  nxt = kernel_utoa(dst, nxt, blockIdx.z);
  nxt = kernel_strcpy(dst, nxt, "), thread (");
  nxt = kernel_utoa(dst, nxt, threadIdx.x);
  nxt = kernel_strcpy(dst, nxt, ",");
  nxt = kernel_utoa(dst, nxt, threadIdx.y);
  nxt = kernel_strcpy(dst, nxt, ",");
  nxt = kernel_utoa(dst, nxt, threadIdx.z);
  nxt = kernel_strcpy(dst, nxt, "), sm ");
  nxt = kernel_utoa(dst, nxt, sm_id);
  nxt = kernel_strcpy(dst, nxt, ", warp ");
  nxt = kernel_utoa(dst, nxt, warp_id);
  nxt = kernel_strcpy(dst, nxt, ", lane ");
  nxt = kernel_utoa(dst, nxt, lane_id);
  nxt = kernel_strcpy(dst, nxt, "] ");

  return nxt;
}

} // namespace

__device__ void kernel_printf_extend_format(char* dst, const char* src) {
    char* dst_base = dst;
    dst = kernel_printf_info(dst);
    dst = kernel_strcpy(dst_base, dst, src);
    *dst = '\0';
}

} // namespace HugeCTR
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

/*
 * This is the header file for a set of extended device printf functions.
 * It usage is almost identical to that of the standard printf function.
 * What it does is to add a prefix to a user provided message, which can be
 * helpful in debugging your kernel without using cuda-gdb everytime.
 * See below for some examples on how to use it.
 *
 * 1. Basic example:
 *
     __global__ void your_kernel(int* dst, int val) {
         ...
         if (threadIdx.x == 0)
           HCTR_KERNEL_PRINT("The used value: %d\n", val);
         ...
     }

 *
 * 1.1. output:
 *
     [HUGECTR][KERNEL][device 0, grid 36, block (427,0,0), thread (0,0,0), sm 39, warp 1, lane 0]
 The used value: 64 [HUGECTR][KERNEL][device 0, grid 36, block (469,0,0), thread (0,0,0), sm 56,
 warp 0, lane 0] The used value: 64 [HUGECTR][KERNEL][device 0, grid 36, block (421,0,0), thread
 (0,0,0), sm 71, warp 0, lane 0] The used value: 64 [HUGECTR][KERNEL][device 0, grid 36, block
 (389,0,0), thread (0,0,0), sm 43, warp 0, lane 0] The used value: 64 [HUGECTR][KERNEL][device 0,
 grid 36, block (445,0,0), thread (0,0,0), sm 71, warp 1, lane 0] The used value: 64
     [HUGECTR][KERNEL][device 0, grid 36, block (417,0,0), thread (0,0,0), sm 9, warp 0, lane 0] The
 used value: 64
 *
 * 2. Conditional printf:
 *
     __global__ void your_kernel(int* dst, int val) {
         ...
        HCTR_KERNEL_PRINT_IF(threadIdx.x == 0, "The used value: %d\n", val);
         ...
     }
 *
 * 3. Printf once per thread block:
 *
     __global__ void your_kernel(int* dst, int val) {
         ...
        HCTR_KERNEL_PRINT_PER_BLOCK("The used value: %d\n", val);
         ...
     }
 *
 * 4. Printf once per grid:
 *
     __global__ void your_kernel(int* dst, int val) {
         ...
        HCTR_KERNEL_PRINT_GRID(threadIdx.x == 0, "The used value: %d\n", val);
         ...
     }
 *
 * Limitations:
 *
 * 1. HCTR_KERNEL_PRINT* can accept at most 7 arguments in addition to the format string.
 * 2. It is not a non-destructive test to insert printfs because its overuse can affect the kernel
 register usage,
 * warp scheduling, the amounts of computation, etc. It is up to users to limit the amount of
 printed messages.
 * It is also recommended that, once you finish debugging your kernel, remove or comment out all the
 printfs.
 * 3. Currently, the maximum length of a printed message is 256 including the prefix (See
 KERNEL_PRINTF_MAX_LENGTH below).
 * If you try printing a longer message, it will be truncated after outputting 256 characters.
 *
 */

#include <base/debug/cuda_debugging.hpp>

namespace HugeCTR {

// The maximum length of printed message. It can affect the SM resource usage and performance to set
// it to a larger value.
const size_t KERNEL_PRINTF_MAX_LENGTH = 256;

/*
 * To directly use the device functions below can lead to an undefined behavior.
 * In addition, their interfaces may change in the future.
 * Use a set of wrapper macros in the bottom instead.
 */
__device__ void kernel_printf_extend_format(char* dst, const char* src);

template <typename Type1>
__device__ void kernel_printf(const char* format, Type1 arg1) {
  char extended_format[KERNEL_PRINTF_MAX_LENGTH] = {};
  kernel_printf_extend_format(extended_format, format);
  printf(extended_format, arg1);
}

template <typename Type1, typename Type2>
__device__ void kernel_printf(const char* format, Type1 arg1, Type2 arg2) {
  char extended_format[KERNEL_PRINTF_MAX_LENGTH] = {};
  kernel_printf_extend_format(extended_format, format);
  printf(extended_format, arg1, arg2);
}

template <typename Type1, typename Type2, typename Type3>
__device__ void kernel_printf(const char* format, Type1 arg1, Type2 arg2, Type3 arg3) {
  char extended_format[KERNEL_PRINTF_MAX_LENGTH] = {};
  kernel_printf_extend_format(extended_format, format);
  printf(extended_format, arg1, arg2, arg3);
}

template <typename Type1, typename Type2, typename Type3, typename Type4>
__device__ void kernel_printf(const char* format, Type1 arg1, Type2 arg2, Type3 arg3, Type4 arg4) {
  char extended_format[KERNEL_PRINTF_MAX_LENGTH] = {};
  kernel_printf_extend_format(extended_format, format);
  printf(extended_format, arg1, arg2, arg3, arg4);
}

template <typename Type1, typename Type2, typename Type3, typename Type4, typename Type5>
__device__ void kernel_printf(const char* format, Type1 arg1, Type2 arg2, Type3 arg3, Type4 arg4,
                              Type5 arg5) {
  char extended_format[KERNEL_PRINTF_MAX_LENGTH] = {};
  kernel_printf_extend_format(extended_format, format);
  printf(extended_format, arg1, arg2, arg3, arg4, arg5);
}

template <typename Type1, typename Type2, typename Type3, typename Type4, typename Type5,
          typename Type6>
__device__ void kernel_printf(const char* format, Type1 arg1, Type2 arg2, Type3 arg3, Type4 arg4,
                              Type5 arg5, Type6 arg6) {
  char extended_format[KERNEL_PRINTF_MAX_LENGTH] = {};
  kernel_printf_extend_format(extended_format, format);
  printf(extended_format, arg1, arg2, arg3, arg4, arg5, arg6);
}

template <typename Type1, typename Type2, typename Type3, typename Type4, typename Type5,
          typename Type6, typename Type7>
__device__ void kernel_printf(const char* format, Type1 arg1, Type2 arg2, Type3 arg3, Type4 arg4,
                              Type5 arg5, Type6 arg6, Type7 arg7) {
  char extended_format[KERNEL_PRINTF_MAX_LENGTH] = {};
  kernel_printf_extend_format(extended_format, format);
  printf(extended_format, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
}

/*
 * They are the macro functions which users should call in their kernels.
 */
#define HCTR_KERNEL_PRINT(...)  \
  do {                          \
    kernel_printf(__VA_ARGS__); \
  } while (0)
#define HCTR_KERNEL_PRINT_IF(cond, ...) \
  do {                                  \
    if (cond) {                         \
      HCTR_KERNEL_PRINT(__VA_ARGS__);   \
    }                                   \
  } while (0)
#define HCTR_KERNEL_PRINT_PER_BLOCK(...)                                                           \
  do {                                                                                             \
    HCTR_KERNEL_PRINT_IF((threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0), __VA_ARGS__); \
  } while (0)
#define HCTR_KERNEL_PRINT_PER_GRID(...)                          \
  do {                                                           \
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { \
      HCTR_KERNEL_PRINT_PER_BLOCK(__VA_ARGS__);                  \
    }                                                            \
  } while (0)

}  // namespace HugeCTR
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
 * This is the header file for a set of host functions to debug kernels.
 * For instance, by using them, you can investigate your kernel's resource usage
 * and execution time withtout going to a profiler everytime.
 * See below for some examples on how to use it.
 *
 * 1. Check your kernels' SM resource usage:
 *
    // Query the occupancy, register usage, etc of 'your_kernel_name' kernel when
    // the thread block size is 256 & per-block shared memory in byte is 1024.
     HCTR_CUDA_KERNEL_SUMMARY(INFO, your_kernel_name, 256, 1024);    // or
     HCTR_CUDA_KERNEL_SUMMARY(WARNING, your_kernel_name, 256, 1024); // or
     HCTR_CUDA_KERNEL_SUMMARY(DEBUG, your_kernel_name, 256, 1024);

 *
 * 2. Check your kernels' average execution time
 *
    // Caculate the execution time of 'your_kernel_name` by launching it 4 times.
    HCTR_CUDA_KERNEL_TIME(INFO, your_kernel_name, 4)    // or
    HCTR_CUDA_KERNEL_TIME(WARNING, your_kernel_name, 4) // or
    HCTR_CUDA_KERNEL_TIME(DEBUG, your_kernel_name, 4)

 *
 */

#include <cuda_runtime.h>

#include <core23/logger.hpp>

#define HCTR_CUDA_KERNEL_SUMMARY(MODE, FUNC, BLOCKSIZE, DSHMEM)                                 \
  do {                                                                                          \
    cudaFuncAttributes func_attr;                                                               \
    int max_blocks = 0;                                                                         \
    HCTR_CUDA_CHECK(ASYNC, cudaFuncGetAttributes(&func_attr, FUNC));                            \
    HCTR_CUDA_CHECK(ASYNC, cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, FUNC,     \
                                                                         BLOCKSIZE, DSHMEM));   \
    HCTR_LOG(MODE, WORLD,                                                                       \
             #FUNC                                                                              \
             " Summary:\n"                                                                      \
             "\tbinaryVersion:             %d\n"                                                \
             "\tcacheModeCA:               %d\n"                                                \
             "\tconstSizeBytes:            %zu\n"                                               \
             "\tlocalSizeBytes:            %zu\n"                                               \
             "\tmaxDynamicSharedSizeBytes: %d\n"                                                \
             "\tmaxThreadsPerBlock:        %d\n"                                                \
             "\tnumRegs:                   %d\n"                                                \
             "\tpreferredShmemCarveout:    %d\n"                                                \
             "\tptxVersion:                %d\n"                                                \
             "\tsharedSizeBytes:           %zu\n"                                               \
             "\tblock size:                %d\n"                                                \
             "\tmax blocks per SM:         %d\n",                                               \
             func_attr.binaryVersion, func_attr.cacheModeCA, func_attr.constSizeBytes,          \
             func_attr.localSizeBytes, func_attr.maxDynamicSharedSizeBytes,                     \
             func_attr.maxThreadsPerBlock, func_attr.numRegs, func_attr.preferredShmemCarveout, \
             func_attr.ptxVersion, func_attr.sharedSizeBytes, BLOCKSIZE, max_blocks);           \
  } while (0)

#define HCTR_CUDA_KERNEL_TIME(MODE, KERNEL_LAUNCH, ITER)              \
  do {                                                                \
    cudaEvent_t st_time, ed_time;                                     \
    cudaEventCreate(&st_time);                                        \
    cudaEventCreate(&ed_time);                                        \
    cudaEventRecord(st_time, stream);                                 \
    for (size_t i = 0; i < ITER; ++i) {                               \
      KERNEL_LAUNCH;                                                  \
    }                                                                 \
    cudaEventRecord(ed_time, stream);                                 \
    cudaEventSynchronize(ed_time);                                    \
    HCTR_CUDA_CHECK(ASYNC, cudaGetLastError());                       \
    float elapsed_ms = 0.0f;                                          \
    cudaEventElapsedTime(&elapsed_ms, st_time, ed_time);              \
    HCTR_LOG(MODE, WORLD,                                             \
             #KERNEL_LAUNCH                                           \
             ":\n"                                                    \
             "\tAvg. (%zu iterations) elapsed time in usec = %.0f\n", \
             ITER, elapsed_ms * 1000 / ITER);                         \
    cudaEventDestroy(st_time);                                        \
    cudaEventDestroy(ed_time);                                        \
  } while (0)

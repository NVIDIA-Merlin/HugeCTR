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

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define HCTR_INLINE __forceinline__
#define HCTR_HOST_DEVICE __host__ __device__
#define HCTR_DEVICE __device__
#define HCTR_HOST __host__
#elif defined(__CUDACC_RTC__)
#define HCTR_INLINE __forceinline__
#define HCTR_HOST_DEVICE __device__
#define HCTR_DEVICE __device__
#define HCTR_HOST
#else
#define HCTR_INLINE inline
#define HCTR_HOST_DEVICE
#define HCTR_DEVICE
#define HCTR_HOST
#endif

// TODO: Add the macros for common CUDA for loops
// TODO: Remove duplicate code and variable
constexpr int kWarpSize = 32;
constexpr int kMaxBlockSize = 1024;
constexpr int64_t kcudaAllocationAlignment = 256;
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

#define CONCAT_IMPL(s1, s2) s1##s2
#define CONCAT(s1, s2) CONCAT_IMPL(s1, s2)
#define ANONYMOUS_VARIABLE(str) CONCAT(str, __LINE__)

#ifdef HCTR_DISALLOW_COPY
#error HCTR_DISALLOW_COPY is already defined. Something is odd. Did you forget/remove "#pragma once" somewhere?
#else
#define HCTR_DISALLOW_COPY(ClassName)   \
  ClassName(const ClassName&) = delete; \
  ClassName& operator=(const ClassName&) = delete
#endif

#ifdef HCTR_DISALLOW_MOVE
#error HCTR_DISALLOW_MOVE is already defined. Something is odd. Did you forget/remove "#pragma once" somewhere?
#else
#define HCTR_DISALLOW_MOVE(ClassName) \
  ClassName(ClassName&&) = delete;    \
  ClassName& operator=(ClassName&&) = delete
#endif

#ifdef HCTR_DISALLOW_COPY_AND_MOVE
#error HCTR_DISALLOW_COPY_AND_MOVE is already defined. Something is odd. Did you forget/remove "#pragma once" somewhere?
#else
#define HCTR_DISALLOW_COPY_AND_MOVE(ClassName) \
  HCTR_DISALLOW_COPY(ClassName);               \
  HCTR_DISALLOW_MOVE(ClassName)
#endif

#define HOST_INLINE __host__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
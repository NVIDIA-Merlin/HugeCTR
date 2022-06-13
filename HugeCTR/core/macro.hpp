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
#define CONCAT_IMPL(s1, s2) s1##s2
#define CONCAT(s1, s2) CONCAT_IMPL(s1, s2)
#define ANONYMOUS_VARIABLE(str) CONCAT(str, __LINE__)

#define DISALLOW_COPY(ClassName)        \
  ClassName(const ClassName&) = delete; \
  ClassName& operator=(const ClassName&) = delete;

#define DISALLOW_MOVE(ClassName)   \
  ClassName(ClassName&&) = delete; \
  ClassName& operator=(ClassName&&) = delete;

#define DISALLOW_COPY_AND_MOVE(ClassName) \
  DISALLOW_COPY(ClassName)                \
  DISALLOW_MOVE(ClassName)

#define HOST_INLINE __host__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
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

#include <core23/cuda_stream.hpp>
#include <core23/curand_generator.hpp>
#include <cstdint>

namespace HugeCTR {

namespace core23 {

class Device;

template <typename Type>
void fill_sync(Type* data, int64_t num_elements, const Type val, const Device& device);

template <typename Type>
void fill_async(Type* data, int64_t num_elements, const Type val, const Device& device,
                CUDAStream stream);

void copy_sync(void* dst_data, const void* src_data, int64_t num_bytes, const Device& dst_device,
               const Device& src_device);
void copy_async(void* dst_data, const void* src_data, int64_t num_bytes, const Device& dst_device,
                const Device& src_device, CUDAStream stream);

template <typename DstType, typename SrcType>
void convert_async(DstType* dst_data, const SrcType* src_data, int64_t num_elements,
                   const Device& dst_device, const Device& src_device, CUDAStream stream);

template <typename Type>
void uniform_async(Type* data, int64_t num_elements, const Type a, const Type b,
                   const Device& device, CURANDGenerator generator, CUDAStream stream);

template <typename Type>
void normal_async(Type* data, int64_t num_elements, const Type mean, const Type stddev,
                  const Device& device, CURANDGenerator generator, CUDAStream stream);

enum class VarianceScalingMode { FanIn, FanOut, FanAvg };

enum class VarianceScalingDistribution { Uniform, TruncatedNormal, UntruncatedNormal };

template <typename Type>
void variance_scaling_async(Type* data, int64_t num_elements, Type scale,
                            const VarianceScalingMode mode,
                            const VarianceScalingDistribution distribution, const Type fan_in,
                            const Type fan_out, const Device& device, CURANDGenerator generator,
                            CUDAStream stream);
template <typename Type>
void xavier_uniform_async(Type* data, int64_t num_elements, const Type fan_in, const Type fan_out,
                          const Device& device, CURANDGenerator generator, CUDAStream stream);

template <typename Type>
void xavier_normal_async(Type* data, int64_t num_elements, const Type fan_in, const Type fan_out,
                         const Device& device, CURANDGenerator generator, CUDAStream stream);

template <typename Type>
void he_uniform_async(Type* data, int64_t num_elements, const Type fan_in, const Device& device,
                      CURANDGenerator generator, CUDAStream stream);

template <typename Type>
void he_normal_async(Type* data, int64_t num_elements, const Type fan_in, const Device& device,
                     CURANDGenerator generator, CUDAStream stream);

}  // namespace core23

}  // namespace HugeCTR
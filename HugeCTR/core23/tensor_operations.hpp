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
#include <core23/tensor.hpp>

namespace HugeCTR {

namespace core23 {

void zeros_sync(Tensor& data);
void zeros_async(Tensor& data, CUDAStream stream = CUDAStream());
void copy_sync(Tensor& dst, const Tensor& src);
void copy_async(Tensor& dst, const Tensor& src, CUDAStream stream = CUDAStream());
void convert_async(Tensor& dst, const Tensor& src, CUDAStream stream = CUDAStream());
void uniform_async(Tensor& data, const float a, const float b, CURANDGenerator generator,
                   CUDAStream stream = CUDAStream());
void normal_async(Tensor& data, const float mean, const float stddev, CURANDGenerator generator,
                  CUDAStream stream = CUDAStream());

}  // namespace core23

}  // namespace HugeCTR
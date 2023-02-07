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

#include <cuda_fp16.h>

#include <core23/data_type.hpp>
#include <core23/macros.hpp>

namespace HugeCTR {
namespace core23 {

#define ALL_DATA_CONVERSIONS_SUPPORTED(PH)    \
  PH(__half, __half, )                        \
  PH(__half, float, __float2half)             \
  PH(float, float, static_cast<float>)        \
  PH(float, __half, __half2float)             \
  PH(float, int32_t, static_cast<float>)      \
  PH(float, int64_t, static_cast<float>)      \
  PH(float, uint32_t, static_cast<float>)     \
  PH(float, uint64_t, static_cast<float>)     \
  PH(double, float, static_cast<double>)      \
  PH(char, float, static_cast<char>)          \
  PH(int8_t, float, static_cast<int8_t>)      \
  PH(int32_t, float, static_cast<int32_t>)    \
  PH(int64_t, float, static_cast<int64_t>)    \
  PH(uint8_t, float, static_cast<uint8_t>)    \
  PH(uint32_t, float, static_cast<uint32_t>)  \
  PH(uint64_t, float, static_cast<uint64_t>)  \
  PH(int32_t, uint32_t, static_cast<int32_t>) \
  PH(int32_t, int64_t, static_cast<int32_t>)

template <typename DstType, typename SrcType>
struct TypeConverter;

#define DEFINE_DATA_TYPE_CONVERTER(DstType, SrcType, EXPR)                               \
  template <>                                                                            \
  struct TypeConverter<DstType, SrcType> {                                               \
    static HCTR_INLINE HCTR_HOST_DEVICE DstType value(SrcType src) { return EXPR(src); } \
  };

ALL_DATA_CONVERSIONS_SUPPORTED(DEFINE_DATA_TYPE_CONVERTER)

}  // namespace core23
}  // namespace HugeCTR
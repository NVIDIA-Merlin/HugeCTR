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

#include <core23/logger.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>

#define CASE_TYPE_USING_HINT_CORE23(enum_type, type, HINT, ...) \
  case (enum_type): {                                           \
    using HINT = type;                                          \
    __VA_ARGS__();                                              \
    break;                                                      \
  }

#define DISPATCH_INTEGRAL_FUNCTION_CORE23(DATA_TYPE, HINT, ...)                          \
  switch (DATA_TYPE) {                                                                   \
    CASE_TYPE_USING_HINT_CORE23(core23::ScalarType::Int64, int64_t, HINT, __VA_ARGS__)   \
    CASE_TYPE_USING_HINT_CORE23(core23::ScalarType::Int32, int32_t, HINT, __VA_ARGS__)   \
    CASE_TYPE_USING_HINT_CORE23(core23::ScalarType::UInt64, uint64_t, HINT, __VA_ARGS__) \
    CASE_TYPE_USING_HINT_CORE23(core23::ScalarType::UInt32, uint32_t, HINT, __VA_ARGS__) \
    default:                                                                             \
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,                                      \
                     "DISPATCH_INTEGRAL_FUNCTION do not support type");                  \
  }

#define DISPATCH_SIGNED_INTEGRAL_FUNCTION_CORE23(DATA_TYPE, HINT, ...)                 \
  switch (DATA_TYPE) {                                                                 \
    CASE_TYPE_USING_HINT_CORE23(core23::ScalarType::Int64, int64_t, HINT, __VA_ARGS__) \
    CASE_TYPE_USING_HINT_CORE23(core23::ScalarType::Int32, int32_t, HINT, __VA_ARGS__) \
    default:                                                                           \
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,                                    \
                     "DISPATCH_SIGNED_INTEGRAL_FUNCTION do not support type");         \
  }

#define DISPATCH_UNSIGNED_INTEGRAL_FUNCTION_CORE23(DATA_TYPE, HINT, ...)                 \
  switch (DATA_TYPE) {                                                                   \
    CASE_TYPE_USING_HINT_CORE23(core23::ScalarType::UInt64, uint64_t, HINT, __VA_ARGS__) \
    CASE_TYPE_USING_HINT_CORE23(core23::ScalarType::UInt32, uint32_t, HINT, __VA_ARGS__) \
    default:                                                                             \
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,                                      \
                     "DISPATCH_UNSIGNED_INTEGRAL_FUNCTION do not support type");         \
  }

#define DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(DATA_TYPE, HINT, ...)                \
  switch (DATA_TYPE) {                                                               \
    CASE_TYPE_USING_HINT_CORE23(core23::ScalarType::Float, float, HINT, __VA_ARGS__) \
    CASE_TYPE_USING_HINT_CORE23(core23::ScalarType::Half, __half, HINT, __VA_ARGS__) \
    default:                                                                         \
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,                                  \
                     "DISPATCH_FLOAT_AND_HALF_FUNCTION do not support type");        \
  }

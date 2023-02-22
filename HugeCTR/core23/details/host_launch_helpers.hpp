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

#include <algorithm>
#include <core23/data_type_helpers.cuh>
#include <cstdint>

namespace HugeCTR {

namespace core23 {

template <typename Type>
struct FillParams {
  FillParams(Type* data, int64_t size, Type val) : data(data), size(size), val(val) {}
  Type* data;
  int64_t size;
  Type val;
};

template <typename DstType, typename SrcType>
struct CopyParams {
  CopyParams(DstType* dst, const SrcType* src, int64_t size) : dst(dst), src(src), size(size) {}
  DstType* dst;
  const SrcType* src;
  int64_t size;
};

template <typename DstType, typename SrcType, typename Op>
struct TransformParams {
  TransformParams(DstType* dst, const SrcType* src, int64_t size, Op op)
      : dst(dst), src(src), size(size), op(op) {}
  DstType* dst;
  const SrcType* src;
  int64_t size;
  Op op;
};

template <typename Type>
void fill_wrapper(void* user_data) {
  FillParams<Type>* params = static_cast<FillParams<Type>*>(user_data);
  std::fill(params->data, params->data + params->size, params->val);
  delete params;
}

void copy_wrapper(void* user_data);

template <typename DstType, typename SrcType, typename Op>
void transform_wrapper(void* user_data) {
  TransformParams<DstType, SrcType, Op>* params =
      static_cast<TransformParams<DstType, SrcType, Op>*>(user_data);
  std::transform(params->src, params->src + params->size, params->dst, params->op);
  delete params;
}

}  // namespace core23
}  // namespace HugeCTR
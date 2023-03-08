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

// clang-format off
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"

#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/embedding/all2all_embedding_collection.hpp"

#include "lookup/impl/core_impl/tf_backend.hpp"
// clang-format on

#include "HugeCTR/core23/device.hpp"
#include "HugeCTR/core23/tensor.hpp"
#include "HugeCTR/core23/device_type.hpp"
#include "HugeCTR/core23/shape.hpp"

namespace sok {
namespace core23 = HugeCTR::core23;
// clang-format off
using CoreResourceManager = ::core::CoreResourceManager;
using TFCoreResourceManager = ::tf_internal::TFCoreResourceManager;
using EmbeddingCollectionParam = ::embedding::EmbeddingCollectionParam;
using UniformModelParallelEmbeddingMeta = ::embedding::UniformModelParallelEmbeddingMeta;
using TablePlacementStrategy   = ::embedding::TablePlacementStrategy;

using Tensor23 = core23::Tensor;
// clang-foramt on

template <typename T>
core23::Tensor convert_tensor_core23(const tensorflow::Tensor* tensor,int device_id) {
  return core23::Tensor::bind(tensor->data(),{tensor->NumElements()},core23::ToScalarType<T>::value,{core23::DeviceType::GPU,device_id});
}

template <typename KeyType, typename OffsetType, typename DType>
std::unique_ptr<::embedding::EmbeddingCollectionParam> make_embedding_collection_param(
                                      const std::vector<std::vector<int>> &shard_matrix,
                                      int num_lookups,
                                      const std::vector<std::string>& combiners,
                                      const std::vector<int>& hotness,
                                      const std::vector<int>& dimensions,
                                      const int global_batch_size,
                                      const int global_gpu_id) {
  const static std::unordered_map<std::string, ::embedding::Combiner> combiner_map = {
    {"sum", ::embedding::Combiner::Sum},
    {"Sum", ::embedding::Combiner::Sum},
    {"SUM", ::embedding::Combiner::Sum},
    {"average", ::embedding::Combiner::Average},
    {"Average", ::embedding::Combiner::Average},
    {"AVERAGE", ::embedding::Combiner::Average},
    {"mean", ::embedding::Combiner::Average},
    {"Mean", ::embedding::Combiner::Average},
    {"MEAN", ::embedding::Combiner::Average}};
  std::vector<::embedding::LookupParam> lookup_params;
  std::vector<int> table_ids(num_lookups);
  std::iota(table_ids.begin(), table_ids.end(), 0);
  for (int i = 0; i < num_lookups; ++i) {
    lookup_params.emplace_back(i, table_ids[i], combiner_map.at(combiners[i]), hotness[i], dimensions[i]);
  }

  
  return std::unique_ptr<::embedding::EmbeddingCollectionParam>( new ::embedding::EmbeddingCollectionParam(
    num_lookups,
    {},
    num_lookups,
    lookup_params,
    shard_matrix,
    {{::embedding::TablePlacementStrategy::ModelParallel, table_ids}},
    global_batch_size,
    core23::ToScalarType<KeyType>::value,
    core23::ToScalarType<int32_t>::value,
    core23::ToScalarType<OffsetType>::value,
    core23::ToScalarType<DType>::value,
    ::embedding::EmbeddingLayout::FeatureMajor,
    ::embedding::EmbeddingLayout::FeatureMajor,
    false
  ));
}
}  // namespace sok

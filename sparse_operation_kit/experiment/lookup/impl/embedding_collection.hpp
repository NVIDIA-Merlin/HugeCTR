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

namespace sok {

// clang-format off
using Tensor              = ::core::Tensor;
using CoreResourceManager = ::core::CoreResourceManager;
using TFCoreResourceManager = ::tf_internal::TFCoreResourceManager;
using EmbeddingCollectionParam = ::embedding::EmbeddingCollectionParam;
using UniformModelParallelEmbeddingMeta = ::embedding::UniformModelParallelEmbeddingMeta;
using TablePlacementStrategy   = ::embedding::TablePlacementStrategy;
// clang-foramt on

template <typename T>
std::shared_ptr<::core::TensorImpl> convert_tensor(const tensorflow::Tensor* tensor) {
  auto storage =
      std::make_shared<::tf_internal::TFStorageWrapper>(tensor->data(), tensor->NumElements());
  std::vector<int64_t> shape{tensor->NumElements()};
  auto impl = std::make_shared<::core::TensorImpl>(storage, 0, shape, ::core::DeviceType::GPU,
                                                   ::HugeCTR::TensorScalarTypeFunc<T>::get_type());
  return impl;
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
    ::HugeCTR::TensorScalarTypeFunc<KeyType>::get_type(),
    ::HugeCTR::TensorScalarTypeFunc<int32_t>::get_type(),
    ::HugeCTR::TensorScalarTypeFunc<OffsetType>::get_type(),
    ::HugeCTR::TensorScalarTypeFunc<DType>::get_type(),
    ::embedding::EmbeddingLayout::FeatureMajor,
    ::embedding::EmbeddingLayout::FeatureMajor,
    false
  ));
}
}  // namespace sok

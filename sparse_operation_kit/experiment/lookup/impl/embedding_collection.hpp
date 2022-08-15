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
using EmbeddingShardParam      = ::embedding::EmbeddingShardParam;
using TablePlacementStrategy   = ::embedding::TablePlacementStrategy;
using ISwizzleKey      = ::embedding::tf::IAll2AllEmbeddingCollectionSwizzleKey;
using SwizzleKey       = ::embedding::tf::All2AllEmbeddingCollectionSwizzleKey;
using IModelForward    = ::embedding::tf::IAll2AllEmbeddingCollectionModelForward;
using ModelForward     = ::embedding::tf::All2AllEmbeddingCollectionModelForward;
using IModelBackward   = ::embedding::tf::IAll2AllEmbeddingCollectionModelBackward;
using ModelBackward    = ::embedding::tf::All2AllEmbeddingCollectionModelBackward;
using INetworkForward  = ::embedding::tf::IAll2AllEmbeddingCollectionNetworkForward;
using NetworkForward   = ::embedding::tf::All2AllEmbeddingCollectionNetworkForward;
using INetworkBackward = ::embedding::tf::IAll2AllEmbeddingCollectionNetworkBackward;
using NetworkBackward  = ::embedding::tf::All2AllEmbeddingCollectionNetworkBackward;
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
void make_embedding_collection_param(::embedding::EmbeddingCollectionParam& ebc_param,
                                     const int num_lookups,
                                     const std::vector<std::string>& combiners,
                                     const std::vector<int>& hotness,
                                     const std::vector<int>& dimensions,
                                     const int global_batch_size) {
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
  // Assuming combiners.size() == hotness.size() == dimensions.size()
  ebc_param.num_embedding = combiners.size();
  for (int i = 0; i < combiners.size(); ++i) {
    ::embedding::EmbeddingParam emb_param;
    emb_param.embedding_id = i;
    emb_param.id_space = i;
    emb_param.combiner = combiner_map.at(combiners[i]);
    emb_param.hotness = hotness[i];
    emb_param.ev_size = dimensions[i];
    ebc_param.embedding_params.push_back(std::move(emb_param));
  }
  ebc_param.universal_batch_size = global_batch_size;
  ebc_param.is_table_first_input = true;
  ebc_param.is_utest = false;
  ebc_param.key_type = ::HugeCTR::TensorScalarTypeFunc<KeyType>::get_type();
  ebc_param.index_type = ::HugeCTR::TensorScalarTypeFunc<int32_t>::get_type();
  ebc_param.offset_type = ::HugeCTR::TensorScalarTypeFunc<OffsetType>::get_type();
  ebc_param.emb_type = ::HugeCTR::TensorScalarTypeFunc<DType>::get_type();
}

}  // namespace sok

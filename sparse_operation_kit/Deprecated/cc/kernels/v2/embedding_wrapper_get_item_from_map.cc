/*
* Copyright (c) 2021, NVIDIA CORPORATION.
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


#include "embedding_wrapper.h"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp"

namespace HugeCTR {
namespace Version2 {

template <typename TypeKey, typename TypeFP>
template <typename Item>
Item EmbeddingWrapper<TypeKey, TypeFP>::get_item_from_map(const std::map<std::string, Item>& map, 
            const std::string& map_key) {
    auto it = map.find(map_key);
    return (it != map.end()) ? it->second : nullptr;
}

template <> 
template <>
std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>
EmbeddingWrapper<long long, float>::get_item_from_map(
    const std::map<std::string, std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>>& map,
    const std::string& map_key) {
    auto it = map.find(map_key);
    return (it != map.end()) ? it->second : std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>();
}
template <>
template <>
std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>
EmbeddingWrapper<unsigned int, float>::get_item_from_map(
    const std::map<std::string, std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>>& map,
    const std::string& map_key) {
    auto it = map.find(map_key);
    return (it != map.end()) ? it->second : std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>();
}
template <>
template <>
std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>
EmbeddingWrapper<long long, __half>::get_item_from_map(
    const std::map<std::string, std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>>& map,
    const std::string& map_key) {
    auto it = map.find(map_key);
    return (it != map.end()) ? it->second : std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>();
}
template <>
template <>
std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>
EmbeddingWrapper<unsigned int, __half>::get_item_from_map(
    const std::map<std::string, std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>>& map,
    const std::string& map_key) {
    auto it = map.find(map_key);
    return (it != map.end()) ? it->second : std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>();
}

template auto EmbeddingWrapper<long long, float>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<InputSpace>>& map,
    const std::string& map_key) -> std::shared_ptr<InputSpace>;
template auto EmbeddingWrapper<long long, __half>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<InputSpace>>& map,
    const std::string& map_key) -> std::shared_ptr<InputSpace>;
template auto EmbeddingWrapper<unsigned int, float>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<InputSpace>>& map,
    const std::string& map_key) -> std::shared_ptr<InputSpace>;
template auto EmbeddingWrapper<unsigned int, __half>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<InputSpace>>& map,
    const std::string& map_key) -> std::shared_ptr<InputSpace>;

template auto EmbeddingWrapper<long long, float>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<IEmbedding>>& map,
    const std::string& map_key) -> std::shared_ptr<IEmbedding>;
template auto EmbeddingWrapper<long long, __half>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<IEmbedding>>& map,
    const std::string& map_key) -> std::shared_ptr<IEmbedding>;
template auto EmbeddingWrapper<unsigned int, float>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<IEmbedding>>& map,
    const std::string& map_key) -> std::shared_ptr<IEmbedding>;
template auto EmbeddingWrapper<unsigned int, __half>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<IEmbedding>>& map,
    const std::string& map_key) -> std::shared_ptr<IEmbedding>;

template auto EmbeddingWrapper<long long, float>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<EmbeddingHyperParams>>& map,
    const std::string& map_key) -> std::shared_ptr<EmbeddingHyperParams>;
template auto EmbeddingWrapper<long long, __half>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<EmbeddingHyperParams>>& map,
    const std::string& map_key) -> std::shared_ptr<EmbeddingHyperParams>;
template auto EmbeddingWrapper<unsigned int, float>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<EmbeddingHyperParams>>& map,
    const std::string& map_key) -> std::shared_ptr<EmbeddingHyperParams>;
template auto EmbeddingWrapper<unsigned int, __half>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<EmbeddingHyperParams>>& map,
    const std::string& map_key) -> std::shared_ptr<EmbeddingHyperParams>;

template <>
template <>
std::vector<cudaEvent_t> EmbeddingWrapper<long long, float>::get_item_from_map(
    const std::map<std::string, std::vector<cudaEvent_t>>& map,
    const std::string& map_key) {
    auto it = map.find(map_key);
    return (it != map.end()) ? it->second : std::vector<cudaEvent_t>();
}
template <>
template <>
std::vector<cudaEvent_t> EmbeddingWrapper<long long, __half>::get_item_from_map(
    const std::map<std::string, std::vector<cudaEvent_t>>& map,
    const std::string& map_key) {
    auto it = map.find(map_key);
    return (it != map.end()) ? it->second : std::vector<cudaEvent_t>();
}
template <>
template <>
std::vector<cudaEvent_t> EmbeddingWrapper<unsigned int, float>::get_item_from_map(
    const std::map<std::string, std::vector<cudaEvent_t>>& map,
    const std::string& map_key) {
    auto it = map.find(map_key);
    return (it != map.end()) ? it->second : std::vector<cudaEvent_t>();
}
template <>
template <>
std::vector<cudaEvent_t> EmbeddingWrapper<unsigned int, __half>::get_item_from_map(
    const std::map<std::string, std::vector<cudaEvent_t>>& map,
    const std::string& map_key) {
    auto it = map.find(map_key);
    return (it != map.end()) ? it->second : std::vector<cudaEvent_t>();
}

template auto EmbeddingWrapper<unsigned int, float>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<DistributeKeysInternelSpaces>>& map, const std::string& map_key)
    -> std::shared_ptr<DistributeKeysInternelSpaces>;
template auto EmbeddingWrapper<long long, float>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<DistributeKeysInternelSpaces>>& map, const std::string& map_key)
    -> std::shared_ptr<DistributeKeysInternelSpaces>;
template auto EmbeddingWrapper<unsigned int, __half>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<DistributeKeysInternelSpaces>>& map, const std::string& map_key)
    -> std::shared_ptr<DistributeKeysInternelSpaces>;
template auto EmbeddingWrapper<long long, __half>::get_item_from_map(
    const std::map<std::string, std::shared_ptr<DistributeKeysInternelSpaces>>& map, const std::string& map_key)
    -> std::shared_ptr<DistributeKeysInternelSpaces>;

template auto EmbeddingWrapper<unsigned int, float>::get_item_from_map(
    const std::map<std::string, distribute_keys_gpu_func_type>& map, const std::string& map_key)
    -> distribute_keys_gpu_func_type;
template auto EmbeddingWrapper<long long, float>::get_item_from_map(
    const std::map<std::string, distribute_keys_gpu_func_type>& map, const std::string& map_key)
    -> distribute_keys_gpu_func_type;
template auto EmbeddingWrapper<unsigned int, __half>::get_item_from_map(
    const std::map<std::string, distribute_keys_gpu_func_type>& map, const std::string& map_key)
    -> distribute_keys_gpu_func_type;
template auto EmbeddingWrapper<long long, __half>::get_item_from_map(
    const std::map<std::string, distribute_keys_gpu_func_type>& map, const std::string& map_key)
    -> distribute_keys_gpu_func_type;

} // namespace Version2
} // namespace HugeCTR
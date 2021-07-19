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
void EmbeddingWrapper<TypeKey, TypeFP>::forward_helper(
    const std::string& embedding_name, const bool is_training) {
    /*get embedding*/
    std::shared_ptr<IEmbedding> embedding = get_item_from_map(embeddings_, embedding_name);
    if (!embedding) {
        const std::string error_info(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " "
                + "Cannot find embedding instance whose name is " + embedding_name);
        throw std::invalid_argument(error_info);
    }

    /*do forward propagation*/
    try {   
        embedding->forward(is_training);
    } catch (const std::exception& error) {
        throw error;
    }
}

template void EmbeddingWrapper<long long, float>::forward_helper(
    const std::string& embedding_name, const bool is_training);
template void EmbeddingWrapper<long long, __half>::forward_helper(
    const std::string& embedding_name, const bool is_training);
template void EmbeddingWrapper<unsigned int, float>::forward_helper(
    const std::string& embedding_name, const bool is_training);
template void EmbeddingWrapper<unsigned int, __half>::forward_helper(
    const std::string& embedding_name, const bool is_training);

} // namespace Version2
} // namespace HugeCTR
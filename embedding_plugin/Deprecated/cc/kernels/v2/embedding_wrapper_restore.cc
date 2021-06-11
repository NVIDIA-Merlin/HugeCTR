/*
* Copyright (c) 2020, NVIDIA CORPORATION.
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
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::restore(const std::string& embedding_name, 
            const std::string& file_name) {
    /*get embedding layer*/
    std::shared_ptr<IEmbedding> embedding = get_item_from_map(embeddings_, embedding_name);
    if (!embedding) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                "Cannot find embedding instance whose name is ", embedding_name);

    /*restore embedding parameters*/
    try {
        embedding->load_parameters(file_name);
    } catch (const std::exception& error) {
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ", error.what());
    }

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::restore(const std::string& embedding_name, 
            const std::string& file_name);
template tensorflow::Status EmbeddingWrapper<long long, __half>::restore(const std::string& embedding_name, 
            const std::string& file_name);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::restore(const std::string& embedding_name, 
            const std::string& file_name);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::restore(const std::string& embedding_name, 
            const std::string& file_name);

} // namespace Version2
} // namespace HugeCTR
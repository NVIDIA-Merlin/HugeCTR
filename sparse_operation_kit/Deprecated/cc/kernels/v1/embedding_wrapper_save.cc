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
#include "embedding_utils.hpp"

namespace HugeCTR {
namespace Version1 {

/**
* This function is used to save embedding table parameters to file.
* @param embedding_name, save which embedding's params to file.
* @param save_name, the name of saved file.
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::save(const std::string& embedding_name, 
                const std::string& save_name) {
    /*get embedding layer*/
    std::shared_ptr<IEmbedding> embedding = get_embedding(embedding_name);
    if (!embedding) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, " Not found ", embedding_name);

    /*save its parameters*/
    try {
        embedding->dump_parameters(save_name);
    } catch (const HugeCTR::internal_runtime_error& rt_err) {
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    rt_err.what());
    } catch (const std::exception& err) {
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    err.what());
    }

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::save(const std::string& embedding_name, 
                const std::string& save_name);
template tensorflow::Status EmbeddingWrapper<long long, __half>::save(const std::string& embedding_name, 
                const std::string& save_name);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::save(const std::string& embedding_name, 
                const std::string& save_name);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::save(const std::string& embedding_name, 
                const std::string& save_name);

} // namespace Version1
} // namespace HugeCTR
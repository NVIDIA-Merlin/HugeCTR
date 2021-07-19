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
#include "embedding_utils.hpp"

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

namespace HugeCTR {
namespace Version1 {


template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::init_embedding_params(const std::string& embedding_name,
        const tensorflow::Tensor* init_value, const bool on_gpu) {
    std::shared_ptr<IEmbedding> embedding = get_embedding(embedding_name);
    if (!embedding) {
        return tensorflow::errors::NotFound("Did not find ", embedding_name, " in embeddings_.");
    }

    try{
        if (init_value->dtype() == tensorflow::DT_BOOL) {
            embedding->init_params();
        } else {
            /*save initial values to file*/
            std::string embedding_file = "TMP_EMBEDDING_INITIAL_VALUES";
            tensorflow::Status status = save_initial_to_file(embedding_name, init_value, embedding_file, on_gpu);
            if (status != tensorflow::Status::OK()) return status;

            /*load initial values to memory*/
            embedding->load_parameters(embedding_file);

            /*delete embedding_file*/
            if (fs::remove_all(embedding_file) == 0)
                return tensorflow::errors::Unavailable(__FILE__, ": ", __LINE__, " Cannot delete ", embedding_file);
        }

    } catch (const HugeCTR::internal_runtime_error& rt_err){
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ", rt_err.what());
    }
    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::init_embedding_params(
        const std::string& embedding_name,
        const tensorflow::Tensor* init_value, const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<long long, __half>::init_embedding_params(
        const std::string& embedding_name,
        const tensorflow::Tensor* init_value, const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::init_embedding_params(
        const std::string& embedding_name,
        const tensorflow::Tensor* init_value, const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::init_embedding_params(
        const std::string& embedding_name,
        const tensorflow::Tensor* init_value, const bool on_gpu);



} // namespace Version1
} // namespace HugeCTR
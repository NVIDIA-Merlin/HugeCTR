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


template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::init_embedding_params(const std::string& embedding_name,
        const tensorflow::Tensor* init_value, const bool on_gpu) {
    std::shared_ptr<IEmbedding> embedding = get_embedding(embedding_name);
    if (!embedding) {
        return tensorflow::errors::NotFound("Did not find ", embedding_name, " in embeddings_.");
    }

    if (init_value->dtype() == tensorflow::DT_BOOL) {
        embedding->init_params();
    } else {
        /*save initial values to file*/
        std::string embedding_file = "TMP_EMBEDDING_INITIAL_VALUES";
        tensorflow::Status status = save_initial_to_file(embedding_name, init_value, embedding_file, on_gpu);
        if (status != tensorflow::Status::OK()) return status;

        /*load initial values to memory*/
        std::ifstream embedding_stream(embedding_file, std::ifstream::binary);
        if (!embedding_stream.is_open()) 
            return tensorflow::errors::Unavailable(__FILE__, ": ", __LINE__, " embedding_stream is not open.");
        embedding->load_parameters(embedding_stream);
        embedding_stream.close();

        /*delete embedding_file*/
        if (std::remove(embedding_file.c_str()) != 0)
            return tensorflow::errors::Unavailable(__FILE__, ": ", __LINE__, " Cannot delete ", embedding_file);
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
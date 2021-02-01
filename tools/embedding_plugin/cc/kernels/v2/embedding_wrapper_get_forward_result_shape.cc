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
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::get_replica_forward_result_shape(
            const std::string& embedding_name, 
            const bool is_training, tensorflow::TensorShape& replica_forward_result_shape) {
    /*get embedding hyper params*/
    std::shared_ptr<EmbeddingHyperParams> hyper_params = get_item_from_map(embedding_hyper_params_, embedding_name);
    if (!hyper_params) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "Cannot find hyper parameters for embedding ", embedding_name);

    /*set forward result shape*/
    size_t gpu_count = resource_manager_->get_local_gpu_count();
    unsigned int replica_batch_size = is_training ? (batch_size_ / gpu_count) : (batch_size_eval_ / gpu_count);
    tensorflow::TensorShape temp_shape = {replica_batch_size, 
                                        hyper_params->slot_num_, 
                                        hyper_params->embedding_vec_size_};

    replica_forward_result_shape = std::move(temp_shape);

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::get_replica_forward_result_shape(
            const std::string& embedding_name, 
            const bool is_training, tensorflow::TensorShape& replica_forward_result_shape);
template tensorflow::Status EmbeddingWrapper<long long, __half>::get_replica_forward_result_shape(
            const std::string& embedding_name, 
            const bool is_training, tensorflow::TensorShape& replica_forward_result_shape);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::get_replica_forward_result_shape(
            const std::string& embedding_name, 
            const bool is_training, tensorflow::TensorShape& replica_forward_result_shape);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::get_replica_forward_result_shape(
            const std::string& embedding_name, 
            const bool is_training, tensorflow::TensorShape& replica_forward_result_shape);

} // namespace Version2
} // namespace HugeCTR
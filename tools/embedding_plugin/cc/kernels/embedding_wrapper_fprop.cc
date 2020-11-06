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



/** This function is used to do forward propagation.
* Will do distribute keys on CPU.
* It is used in plugin.fprop.
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::fprop(const tensorflow::Tensor* sparse_indices, 
                        const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
                        const std::string& embedding_name, const bool is_training,
                        tensorflow::Tensor* const forward_result, const bool on_gpu) {
    tensorflow::Status status;

    /*get embedding params for this instance*/
    std::shared_ptr<EmbeddingParams> params = get_embedding_params(embedding_name);
    if (!params) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, " Not found embedding params for ", embedding_name);

    /*distribute keys to each GPU*/
    status = distribute_keys(sparse_indices, values, dense_shape, embedding_name, is_training, 
                             params->embedding_type_, on_gpu);
    if (status != tensorflow::Status::OK()) return status;

    /*forward propagation*/
    std::shared_ptr<IEmbedding> embedding = get_embedding(embedding_name);
    if (!embedding) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, " Not found ", embedding_name);
    embedding->forward(is_training);

    /*get forward results*/
    if (std::is_same<TypeFP, float>::value) {
        embedding->get_forward_results_tf(is_training, on_gpu, 
                        reinterpret_cast<void*>(forward_result->flat<float>().data()));
    } else if (std::is_same<TypeFP, __half>::value) {
        embedding->get_forward_results_tf(is_training, on_gpu, 
                        reinterpret_cast<void*>(forward_result->flat<Eigen::half>().data()));
    } else {
        return tensorflow::errors::Unimplemented(__FILE__, ": ", __LINE__, " TypeFP should be {float, __half}.");
    }

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::fprop(
                        const tensorflow::Tensor* sparse_indices, 
                        const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
                        const std::string& embedding_name, const bool is_training,
                        tensorflow::Tensor* const forward_result, const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<long long, __half>::fprop(
                        const tensorflow::Tensor* sparse_indices, 
                        const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
                        const std::string& embedding_name, const bool is_training,
                        tensorflow::Tensor* const forward_result, const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::fprop(
                        const tensorflow::Tensor* sparse_indices, 
                        const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
                        const std::string& embedding_name, const bool is_training,
                        tensorflow::Tensor* const forward_result, const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::fprop(
                        const tensorflow::Tensor* sparse_indices, 
                        const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
                        const std::string& embedding_name, const bool is_training,
                        tensorflow::Tensor* const forward_result, const bool on_gpu);

} // namespace Version1
} // namespace HugeCTR
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

namespace HugeCTR {
namespace Version1 {



/** This function is used to get the shape of output tensor. (All GPUs.)
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::get_output_tensor_shape(const std::string& embedding_name, 
                                                const bool is_training,
                                                tensorflow::TensorShape& shape) {
    /*get embedding params for this instance*/
    std::shared_ptr<EmbeddingParams> params = get_embedding_params(embedding_name);
    if (!params) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, " Not found embedding params for ", embedding_name);

    /*set shape*/
    tensorflow::TensorShape tempshape = {(is_training ? batch_size_ : batch_size_eval_), 
                                         params->slot_num_, params->embedding_vec_size_};
    shape = std::move(tempshape);

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::get_output_tensor_shape(
                const std::string& embedding_name, const bool is_training,
                tensorflow::TensorShape& shape);
template tensorflow::Status EmbeddingWrapper<long long, __half>::get_output_tensor_shape(
                const std::string& embedding_name, const bool is_training,
                tensorflow::TensorShape& shape);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::get_output_tensor_shape(
                const std::string& embedding_name, const bool is_training,
                tensorflow::TensorShape& shape);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::get_output_tensor_shape(
                const std::string& embedding_name, const bool is_training,
                tensorflow::TensorShape& shape);

} // namespace Version1
} // namespace HugeCTR
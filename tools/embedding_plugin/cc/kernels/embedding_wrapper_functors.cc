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
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::DoDistributedDistributeKeysFunctor::operator()(
                    EmbeddingWrapper<TypeKey, TypeFP>* const wrapper,
                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                    const tensorflow::Tensor* input_keys,
                    std::vector<tensorflow::Tensor*> row_offset_output,
                    std::vector<tensorflow::Tensor*> value_tensor_output,
                    tensorflow::Tensor* nnz_array_output) {
    return wrapper->distributed_embedding_distribute_keys_helper(distribute_keys_space, input_keys,
                                                        row_offset_output, value_tensor_output, 
                                                        nnz_array_output);
}

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::DoLocalizedDistributeKeysFunctor::operator()(
                    EmbeddingWrapper<TypeKey, TypeFP>* const wrapper,
                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                    const tensorflow::Tensor* input_keys,
                    std::vector<tensorflow::Tensor*> row_offset_output,
                    std::vector<tensorflow::Tensor*> value_tensor_output,
                    tensorflow::Tensor* nnz_array_output) {
    return tensorflow::errors::Unimplemented(__FILE__, ":", __LINE__, " Not implemented yet.");                    
}

template tensorflow::Status EmbeddingWrapper<long long, float>::DoDistributedDistributeKeysFunctor::operator()(
                    EmbeddingWrapper<long long, float>* const wrapper,
                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                    const tensorflow::Tensor* input_keys,
                    std::vector<tensorflow::Tensor*> row_offset_output,
                    std::vector<tensorflow::Tensor*> value_tensor_output,
                    tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<long long, __half>::DoDistributedDistributeKeysFunctor::operator()(
                    EmbeddingWrapper<long long, __half>* const wrapper,
                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                    const tensorflow::Tensor* input_keys,
                    std::vector<tensorflow::Tensor*> row_offset_output,
                    std::vector<tensorflow::Tensor*> value_tensor_output,
                    tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::DoDistributedDistributeKeysFunctor::operator()(
                    EmbeddingWrapper<unsigned int, float>* const wrapper,
                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                    const tensorflow::Tensor* input_keys,
                    std::vector<tensorflow::Tensor*> row_offset_output,
                    std::vector<tensorflow::Tensor*> value_tensor_output,
                    tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::DoDistributedDistributeKeysFunctor::operator()(
                    EmbeddingWrapper<unsigned int, __half>* const wrapper,
                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                    const tensorflow::Tensor* input_keys,
                    std::vector<tensorflow::Tensor*> row_offset_output,
                    std::vector<tensorflow::Tensor*> value_tensor_output,
                    tensorflow::Tensor* nnz_array_output);

template tensorflow::Status EmbeddingWrapper<long long, float>::DoLocalizedDistributeKeysFunctor::operator()(
                    EmbeddingWrapper<long long, float>* const wrapper,
                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                    const tensorflow::Tensor* input_keys,
                    std::vector<tensorflow::Tensor*> row_offset_output,
                    std::vector<tensorflow::Tensor*> value_tensor_output,
                    tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<long long, __half>::DoLocalizedDistributeKeysFunctor::operator()(
                    EmbeddingWrapper<long long, __half>* const wrapper,
                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                    const tensorflow::Tensor* input_keys,
                    std::vector<tensorflow::Tensor*> row_offset_output,
                    std::vector<tensorflow::Tensor*> value_tensor_output,
                    tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::DoLocalizedDistributeKeysFunctor::operator()(
                    EmbeddingWrapper<unsigned int, float>* const wrapper,
                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                    const tensorflow::Tensor* input_keys,
                    std::vector<tensorflow::Tensor*> row_offset_output,
                    std::vector<tensorflow::Tensor*> value_tensor_output,
                    tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::DoLocalizedDistributeKeysFunctor::operator()(
                    EmbeddingWrapper<unsigned int, __half>* const wrapper,
                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                    const tensorflow::Tensor* input_keys,
                    std::vector<tensorflow::Tensor*> row_offset_output,
                    std::vector<tensorflow::Tensor*> value_tensor_output,
                    tensorflow::Tensor* nnz_array_output);

} // namespace Version1
} // namespace HugeCTR
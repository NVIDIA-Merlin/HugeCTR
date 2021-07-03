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
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::input_space_memory_reset(
    const std::string& embedding_name, const int replica_id, const bool is_training) {
    /*get input space*/
    const std::string embedding_input_name = is_training ? (embedding_name + "_train") : (embedding_name + "_eval");
    auto input_spaces = get_item_from_map(input_spaces_, embedding_input_name);
    if (!input_spaces) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                "Cannot find input_spaces for ", embedding_name);

    /*memory reset*/
    auto stream = resource_manager_->get_local_gpu(replica_id)->get_stream();
    WRAPPER_CUDA_CHECK(cudaMemsetAsync(input_spaces->row_offsets_tensors_[replica_id].get_ptr(),
                                       0, 
                                       input_spaces->row_offsets_tensors_[replica_id].get_size_in_bytes(),
                                       stream));
    WRAPPER_CUDA_CHECK(cudaMemsetAsync(input_spaces->value_tensors_[replica_id].get_ptr(),
                                       0,
                                       input_spaces->value_tensors_[replica_id].get_size_in_bytes(),
                                       stream));

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::input_space_memory_reset(
    const std::string& embedding_name, const int replica_id, const bool is_training);
template tensorflow::Status EmbeddingWrapper<long long, __half>::input_space_memory_reset(
    const std::string& embedding_name, const int replica_id, const bool is_training);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::input_space_memory_reset(
    const std::string& embedding_name, const int replica_id, const bool is_training);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::input_space_memory_reset(
    const std::string& embedding_name, const int replica_id, const bool is_training);

} // namespace Version2
} // namespace HugeCTR
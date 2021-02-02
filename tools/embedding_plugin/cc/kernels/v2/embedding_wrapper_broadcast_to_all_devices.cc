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

#ifdef PLUGIN_NVTX
#include <nvToolsExt.h> 
#endif

namespace HugeCTR {
namespace Version2 { 

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::broadcast_then_convert_to_CSR(
            const tensorflow::Tensor* row_indices,
            const tensorflow::Tensor* values, const std::string& embedding_name, 
            const bool is_training, const cudaStream_t& tf_stream) {
#ifdef PLUGIN_NVTX
    nvtxRangeId_t broadcast_id = nvtxRangeStartA("broadcast_&_convert_to_CSR");
#endif

    /*get input space*/
    std::string input_space_name = embedding_name;
    input_space_name += (is_training ? "_train" : "_eval");
    std::shared_ptr<InputSpace> space = get_item_from_map(input_spaces_, input_space_name);
    if (!space) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, ", Did not find ", 
                                                    input_space_name, " in input_spaces.");

    /*do distribute keys*/
    const auto distribute_keys_func = get_item_from_map(distribute_keys_on_gpu_func_, embedding_name);
    WRAPPER_REQUIRE_OK((this->*distribute_keys_func)(row_indices, values, embedding_name, is_training, space));

    /*synchronize with tf stream*/
    std::vector<cudaEvent_t> to_csr_events = get_item_from_map(to_csr_events_, embedding_name);
    if (to_csr_events.empty()) return tensorflow::errors::NotFound(__FILE__, ":", __LINE__, " ",
                        "Did not find to_csr cudaEvent_t for embedding: ", embedding_name);
    for (auto& event : to_csr_events) {
        WRAPPER_CUDA_CHECK(cudaStreamWaitEvent(tf_stream, event, 0));
    }

#ifdef PLUGIN_NVTX
    nvtxRangeEnd(broadcast_id);
#endif
    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::broadcast_then_convert_to_CSR(
            const tensorflow::Tensor* row_indices,
            const tensorflow::Tensor* values, const std::string& embedding_name, 
            const bool is_training, const cudaStream_t& tf_stream);
template tensorflow::Status EmbeddingWrapper<long long, __half>::broadcast_then_convert_to_CSR(
            const tensorflow::Tensor* row_indices,
            const tensorflow::Tensor* values, const std::string& embedding_name, 
            const bool is_training, const cudaStream_t& tf_stream);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::broadcast_then_convert_to_CSR(
            const tensorflow::Tensor* row_indices,
            const tensorflow::Tensor* values, const std::string& embedding_name, 
            const bool is_training, const cudaStream_t& tf_stream);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::broadcast_then_convert_to_CSR(
            const tensorflow::Tensor* row_indices,
            const tensorflow::Tensor* values, const std::string& embedding_name, 
            const bool is_training, const cudaStream_t& tf_stream);

} // namespace Version2
} // namespace HugeCTR
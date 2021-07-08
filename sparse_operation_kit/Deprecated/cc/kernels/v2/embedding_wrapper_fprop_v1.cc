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

#ifdef PLUGIN_NVTX
#include <nvToolsExt.h> 
#endif

namespace HugeCTR {
namespace Version2 {

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::fprop_v1(const std::string& embedding_name, 
            const bool is_training, const tensorflow::Tensor* replica_id, 
            const tensorflow::Tensor* row_offset, const tensorflow::Tensor* values, 
            const tensorflow::Tensor* nnz, const cudaStream_t& tf_stream, 
            const bool input_buffer_reset,
            tensorflow::Tensor* replica_forward_result) {
#ifdef PLUGIN_NVTX
    nvtxRangeId_t fprop_id = nvtxRangeStartA("plugin_fprop");
#endif

    /*copy inputs to embedding input buffer*/
    int host_replica_id = 0;
    WRAPPER_REQUIRE_OK(copy_to_input_space(embedding_name, is_training, replica_id, row_offset,
                        values, nnz, tf_stream, input_buffer_reset, host_replica_id));

    /*do forward propagation once*/
    try {
        call_once(&EmbeddingWrapper<TypeKey, TypeFP>::forward_helper, this, embedding_name, is_training);
    } catch (const std::exception& error) {
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                error.what());
    }

    /*get forward result*/
    WRAPPER_REQUIRE_OK(copy_from_output_tensor(embedding_name, host_replica_id, is_training, replica_forward_result));

    /*synchronize plugin stream with tf stream*/
    std::vector<cudaEvent_t> fprop_events = get_item_from_map(fprop_events_, embedding_name);
    if (fprop_events.empty()) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "Cannot find fprop cudaEvent_t for embedding: ", embedding_name);
    WRAPPER_CUDA_CHECK(cudaStreamWaitEvent(tf_stream, fprop_events[host_replica_id], 0));

    /*reset memory of input space*/
    if (input_buffer_reset) { WRAPPER_REQUIRE_OK(input_space_memory_reset(embedding_name, host_replica_id, is_training)); }
    
#ifdef PLUGIN_NVTX
    nvtxRangeEnd(fprop_id);
#endif

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::fprop_v1(const std::string& embedding_name, 
            const bool is_training, const tensorflow::Tensor* replica_id, 
            const tensorflow::Tensor* row_offset, const tensorflow::Tensor* values, 
            const tensorflow::Tensor* nnz, const cudaStream_t& tf_stream, 
            const bool input_buffer_reset,
            tensorflow::Tensor* replica_forward_result);
template tensorflow::Status EmbeddingWrapper<long long, __half>::fprop_v1(const std::string& embedding_name, 
            const bool is_training, const tensorflow::Tensor* replica_id, 
            const tensorflow::Tensor* row_offset, const tensorflow::Tensor* values, 
            const tensorflow::Tensor* nnz, const cudaStream_t& tf_stream, 
            const bool input_buffer_reset,
            tensorflow::Tensor* replica_forward_result);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::fprop_v1(const std::string& embedding_name, 
            const bool is_training, const tensorflow::Tensor* replica_id, 
            const tensorflow::Tensor* row_offset, const tensorflow::Tensor* values, 
            const tensorflow::Tensor* nnz, const cudaStream_t& tf_stream, 
            const bool input_buffer_reset,
            tensorflow::Tensor* replica_forward_result);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::fprop_v1(const std::string& embedding_name, 
            const bool is_training, const tensorflow::Tensor* replica_id, 
            const tensorflow::Tensor* row_offset, const tensorflow::Tensor* values, 
            const tensorflow::Tensor* nnz, const cudaStream_t& tf_stream, 
            const bool input_buffer_reset,
            tensorflow::Tensor* replica_forward_result);

} // namespace Version2
} // namespace HugeCTR
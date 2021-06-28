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
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::bprop(const std::string& embedding_name, 
            const tensorflow::Tensor* replica_id, const tensorflow::Tensor* replica_top_gradients, 
            const cudaStream_t& tf_stream) {
#ifdef PLUGIN_NVTX
    nvtxRangeId_t bprop_id = nvtxRangeStartA("plugin_bprop");
#endif

    /*copy top_gradients to output tensor buffer*/
    int host_replica_id = 0;
    WRAPPER_REQUIRE_OK(copy_grads_to_output_tensor(embedding_name, replica_id, replica_top_gradients,
                        tf_stream, host_replica_id));

    /*do backward propagation and update params*/
    try {
        call_once(&EmbeddingWrapper<TypeKey, TypeFP>::backward_helper, this, embedding_name);
    } catch (const std::exception& error) {
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                error.what());
    }

    /*synchronize plugin stream with tf stream*/
    std::vector<cudaEvent_t> bprop_events = get_item_from_map(bprop_events_, embedding_name);
    if (bprop_events.empty()) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "Cannot find fprop cudaEvent_t for embedding: ", embedding_name);
    WRAPPER_CUDA_CHECK(cudaStreamWaitEvent(tf_stream, bprop_events[host_replica_id], 0));

#ifdef PLUGIN_NVTX
    nvtxRangeEnd(bprop_id);
#endif

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::bprop(const std::string& embedding_name, 
            const tensorflow::Tensor* replica_id, const tensorflow::Tensor* replica_top_gradients, 
            const cudaStream_t& tf_stream);
template tensorflow::Status EmbeddingWrapper<long long, __half>::bprop(const std::string& embedding_name, 
            const tensorflow::Tensor* replica_id, const tensorflow::Tensor* replica_top_gradients, 
            const cudaStream_t& tf_stream);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::bprop(const std::string& embedding_name, 
            const tensorflow::Tensor* replica_id, const tensorflow::Tensor* replica_top_gradients, 
            const cudaStream_t& tf_stream);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::bprop(const std::string& embedding_name, 
            const tensorflow::Tensor* replica_id, const tensorflow::Tensor* replica_top_gradients, 
            const cudaStream_t& tf_stream);

} // namespace Version2
} // namespace HugeCTR
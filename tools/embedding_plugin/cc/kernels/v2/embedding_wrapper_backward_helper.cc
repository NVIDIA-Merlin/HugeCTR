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
void EmbeddingWrapper<TypeKey, TypeFP>::backward_helper(const std::string& embedding_name) {
    /*get embedding*/
    std::shared_ptr<IEmbedding> embedding = get_item_from_map(embeddings_, embedding_name);
    if (!embedding) {
        const std::string error_info(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " "
                + "Cannot find embedding instance whose name is " + embedding_name);
        throw std::invalid_argument(error_info);
    }

    /*do backward propagation and update params*/
    try {
        embedding->backward();
        embedding->update_params();
    } catch (const std::exception& error) {
        throw error;
    }

    /*record bprop cudaEvent on each device*/
    std::vector<cudaEvent_t> bprop_events = get_item_from_map(bprop_events_, embedding_name);
    if (bprop_events.empty()) {
        const std::string error_info(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " "
                + "Cannot find bprop cudaEvent_t for embedding: " + embedding_name);
        throw std::invalid_argument(error_info);
    }
    for (size_t dev_id = 0; dev_id < resource_manager_->get_local_gpu_count(); ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        CudaDeviceContext context(local_gpu->get_device_id());

        cudaError_t re = cudaEventRecord(bprop_events[dev_id], local_gpu->get_stream());
        if (cudaSuccess != re) {
            const std::string error_info(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " "
                + cudaGetErrorString(re));
            throw std::invalid_argument(error_info);
        }
    } // for dev_id
    
}

template void EmbeddingWrapper<long long, float>::backward_helper(const std::string& embedding_name);
template void EmbeddingWrapper<long long, __half>::backward_helper(const std::string& embedding_name);
template void EmbeddingWrapper<unsigned int, float>::backward_helper(const std::string& embedding_name);
template void EmbeddingWrapper<unsigned int, __half>::backward_helper(const std::string& embedding_name);

} // namespace Version2
} // namespace HugeCTR
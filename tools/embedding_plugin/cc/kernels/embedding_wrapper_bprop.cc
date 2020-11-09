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

/** This function is used to do back propagation and update embedding parameters
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::bprop(const std::string& embedding_name, 
                                                            const tensorflow::Tensor* top_gradients,
                                                            const bool on_gpu) {
    /*get embedding instance*/
    std::shared_ptr<IEmbedding> embedding = get_embedding(embedding_name);
    if (!embedding) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, " Not found ", embedding_name);

    /*distribute top_gradients to each GPU's output tensor*/
    if (std::is_same<TypeFP, float>::value) {
        WRAPPER_CUDA_CHECK(embedding->update_top_gradients(on_gpu, 
                            reinterpret_cast<const void*>(top_gradients->flat<float>().data())));
    } else if (std::is_same<TypeFP, __half>::value) {
        WRAPPER_CUDA_CHECK(embedding->update_top_gradients(on_gpu, 
                            reinterpret_cast<const void*>(top_gradients->flat<Eigen::half>().data())));
    } else {
        return tensorflow::errors::Unimplemented(__FILE__, ":", __LINE__, " TypeFP should be {float, __half}.");
    } 

    /*back propagation and update params*/
    embedding->backward();
    embedding->update_params();

    /*sync each GPU*/
    for (size_t dev_id = 0; dev_id < resource_manager_->get_local_gpu_count(); ++dev_id){
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        CudaDeviceContext context(local_gpu->get_device_id());
        WRAPPER_CUDA_CHECK(cudaStreamSynchronize(local_gpu->get_stream()));
    }

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::bprop(
                    const std::string& embedding_name, const tensorflow::Tensor* top_gradients,
                    const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<long long, __half>::bprop(
                    const std::string& embedding_name, const tensorflow::Tensor* top_gradients,
                    const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::bprop(
                    const std::string& embedding_name, const tensorflow::Tensor* top_gradients,
                    const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::bprop(
                    const std::string& embedding_name, const tensorflow::Tensor* top_gradients,
                    const bool on_gpu);
                    
} // namespace Version1
} // HugeCTR

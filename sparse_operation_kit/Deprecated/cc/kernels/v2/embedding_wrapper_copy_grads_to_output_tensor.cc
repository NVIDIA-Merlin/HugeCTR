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
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::copy_grads_to_output_tensor(
                const std::string& embedding_name, const tensorflow::Tensor* replica_id, 
                const tensorflow::Tensor* replica_top_gradients, const cudaStream_t& tf_stream, 
                int& host_replica_id) {
    /*copy replica_id from GPU to CPU*/
    auto replica_id_flat = replica_id->flat<int>();
    if (replica_id_flat.size() != 1) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                "replica_id should be a scaler.");
    host_replica_id = 0;
    WRAPPER_CUDA_CHECK(cudaMemcpyAsync(&host_replica_id, replica_id_flat.data(), 
                sizeof(int) * 1, cudaMemcpyDeviceToHost, tf_stream));
    WRAPPER_CUDA_CHECK(cudaStreamSynchronize(tf_stream));
    size_t gpu_count = resource_manager_->get_local_gpu_count();
    if (host_replica_id < 0 || host_replica_id >= static_cast<long long>(gpu_count)) 
                return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, 
                " replica_id should be in range of [0, gpu_count)");

    /*get output tensor*/
    std::shared_ptr<IEmbedding> embedding = get_item_from_map(embeddings_, embedding_name);
    if (!embedding) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                "Cannot find embedding instance whose name is ", embedding_name);
    std::vector<TensorBag2> output_tensors_bag = embedding->get_train_output_tensors();
    HugeCTR::Tensor2<TypeFP> replica_output_tensor = 
                HugeCTR::Tensor2<TypeFP>::stretch_from(output_tensors_bag[host_replica_id]);

    /*copy to output tensor buffer*/
    auto replica_top_gradients_flat = replica_top_gradients->flat<float>(); // TODO: Eigen::half or float
    if (replica_top_gradients_flat.size() != static_cast<long long>(replica_output_tensor.get_num_elements())) 
                return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                "the shape of this op's top_gradients is not equal to that of plugin's ",
                "corresponding output tensor.");

    auto local_gpu = resource_manager_->get_local_gpu(host_replica_id);
    auto stream = local_gpu->get_stream();
    WRAPPER_CUDA_CHECK(cudaMemcpyAsync(replica_output_tensor.get_ptr(),
                                        replica_top_gradients_flat.data(),
                                        replica_output_tensor.get_size_in_bytes(),
                                        cudaMemcpyDeviceToDevice,
                                        stream));
                                        
    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::copy_grads_to_output_tensor(
                const std::string& embedding_name, const tensorflow::Tensor* replica_id, 
                const tensorflow::Tensor* replica_top_gradients, const cudaStream_t& tf_stream, 
                int& host_replica_id);
template tensorflow::Status EmbeddingWrapper<long long, __half>::copy_grads_to_output_tensor(
                const std::string& embedding_name, const tensorflow::Tensor* replica_id, 
                const tensorflow::Tensor* replica_top_gradients, const cudaStream_t& tf_stream, 
                int& host_replica_id);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::copy_grads_to_output_tensor(
                const std::string& embedding_name, const tensorflow::Tensor* replica_id, 
                const tensorflow::Tensor* replica_top_gradients, const cudaStream_t& tf_stream, 
                int& host_replica_id);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::copy_grads_to_output_tensor(
                const std::string& embedding_name, const tensorflow::Tensor* replica_id, 
                const tensorflow::Tensor* replica_top_gradients, const cudaStream_t& tf_stream, 
                int& host_replica_id);

} // namespace Version2
} // namespace HugeCTR
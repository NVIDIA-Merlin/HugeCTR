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
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::copy_from_output_tensor(
                const std::string& embedding_name, const int replica_id,
                const bool is_training, tensorflow::Tensor* replica_forward_result) {
    std::shared_ptr<IEmbedding> embedding = get_item_from_map(embeddings_, embedding_name);
    if (!embedding) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "Cannot find embedding instance whose name is ", embedding_name);

    std::vector<TensorBag2> output_tensors_bag = is_training
                                            ? embedding->get_train_output_tensors()
                                            : embedding->get_evaluate_output_tensors();
    size_t gpu_count = resource_manager_->get_local_gpu_count();
    if (replica_id < 0 || replica_id >= static_cast<long long>(gpu_count)) return tensorflow::errors::Aborted(
                    __FILE__, ":", __LINE__, " replica_id should be in range of [0, gpu_count)");
    HugeCTR::Tensor2<TypeFP> replica_output_tensor = 
            HugeCTR::Tensor2<TypeFP>::stretch_from(output_tensors_bag[replica_id]);

    auto replica_forward_result_flat = replica_forward_result->flat<float>();
    if (replica_forward_result_flat.size() != static_cast<long long>(replica_output_tensor.get_num_elements())) 
            return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " the shape of this op's output ",
                    "is not equal to that of plugin's corresponding output tensor.");

    auto local_gpu = resource_manager_->get_local_gpu(replica_id);
    auto stream = local_gpu->get_stream();
    CudaDeviceContext context(local_gpu->get_device_id()); // TODO: necessary??
    WRAPPER_CUDA_CHECK(cudaMemcpyAsync(replica_forward_result_flat.data(),
                                        replica_output_tensor.get_ptr(),
                                        replica_output_tensor.get_size_in_bytes(),
                                        cudaMemcpyDeviceToDevice,
                                        stream));

    /*record event on this stream*/
    std::vector<cudaEvent_t> fprop_events = get_item_from_map(fprop_events_, embedding_name);
    if (fprop_events.empty()) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "Cannot find fprop cudaEvent_t for embedding: ", embedding_name);
    WRAPPER_CUDA_CHECK(cudaEventRecord(fprop_events[replica_id], stream));

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::copy_from_output_tensor(
                const std::string& embedding_name, const int replica_id,
                const bool is_training, tensorflow::Tensor* replica_forward_result);
template tensorflow::Status EmbeddingWrapper<long long, __half>::copy_from_output_tensor(
                const std::string& embedding_name, const int replica_id,
                const bool is_training, tensorflow::Tensor* replica_forward_result);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::copy_from_output_tensor(
                const std::string& embedding_name, const int replica_id,
                const bool is_training, tensorflow::Tensor* replica_forward_result);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::copy_from_output_tensor(
                const std::string& embedding_name, const int replica_id,
                const bool is_training, tensorflow::Tensor* replica_forward_result);

} // namespace Version2
} // namespace HugeCTR
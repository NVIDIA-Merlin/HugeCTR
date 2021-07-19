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


/** This function is used to do forward propagation, and keys have been distributed to each GPU.
* just need to copy those temp CSR to embedding instance's CSR.
* Its inputs: row_offsets, value_tensors are list of tensors.
* TODO: Use NCCL to do the memory transfer, might get better performance.
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::fprop_v2(const tensorflow::OpInputList& row_offsets, 
                            const tensorflow::OpInputList& value_tensors,
                            const tensorflow::Tensor* nnz_array, const std::string& embedding_name, 
                            const bool is_training, tensorflow::Tensor* const forward_result) {
    tensorflow::Status status;

    size_t gpu_count = resource_manager_->get_local_gpu_count();
    if (static_cast<size_t>(row_offsets.size()) != gpu_count)
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                                           "Length of list of row_offsets & value_tensors is less than gpu_count.");

    /*get input space*/
    std::string input_space_name = embedding_name;
    input_space_name += (is_training ? "_train" : "_eval");
    std::shared_ptr<InputSpace> space = get_input_space(input_space_name);
    if (!space) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, ", Did not find ", 
                                                    input_space_name, " in input_spaces.");

    /*copy to each GPU's input space*/
    HugeCTR::CudaDeviceContext context;
    std::unique_ptr<long long []> host_nnz_array(new long long[gpu_count]());
    auto nnz_array_flat = nnz_array->flat<TypeKey>();
    WRAPPER_CUDA_CHECK(cudaMemcpy(host_nnz_array.get(), nnz_array_flat.data(),
                                sizeof(long long) * nnz_array_flat.size(),
                                cudaMemcpyDeviceToHost));

    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        int cur_device = resource_manager_->get_local_gpu(dev_id)->get_device_id();
        context.set_device(cur_device);

        auto row_offset_flat = row_offsets[dev_id].flat<TypeKey>();
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(space->row_offsets_tensors_[dev_id].get_ptr(),
                                            row_offset_flat.data(),
                                            space->row_offsets_tensors_[dev_id].get_size_in_bytes(),
                                            cudaMemcpyDeviceToDevice,
                                            resource_manager_->get_local_gpu(dev_id)->get_stream()));

        auto value_tensor_flat = value_tensors[dev_id].flat<TypeKey>();
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(space->value_tensors_[dev_id].get_ptr(),
                                            value_tensor_flat.data(),
                                            space->value_tensors_[dev_id].get_size_in_bytes(),
                                            cudaMemcpyDeviceToDevice,
                                            resource_manager_->get_local_gpu(dev_id)->get_stream()));

        /*write nnz buffer*/
        *(space->nnz_array_[dev_id]) = static_cast<size_t>(host_nnz_array[dev_id]);
    }

    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        int cur_device = resource_manager_->get_local_gpu(dev_id)->get_device_id();
        context.set_device(cur_device);

        WRAPPER_CUDA_CHECK(cudaStreamSynchronize(resource_manager_->get_local_gpu(dev_id)->get_stream()));
    }

    /*forward propagation*/
    std::shared_ptr<IEmbedding> embedding = get_embedding(embedding_name);
    if (!embedding) return tensorflow::errors::NotFound(__FILE__, ":", __LINE__, " ",
                                    "Not found ", embedding_name);
    embedding->forward(is_training);

    /*get forward results*/
    if (std::is_same<TypeFP, float>::value) {
        embedding->get_forward_results_tf(is_training, true, reinterpret_cast<void*>(forward_result->flat<float>().data()));
    } else if (std::is_same<TypeFP, __half>::value) {
        embedding->get_forward_results_tf(is_training, true, reinterpret_cast<void*>(forward_result->flat<Eigen::half>().data()));
    } else {
        return tensorflow::errors::Unimplemented(__FILE__, ":", __LINE__, " TypeFP should be {float, __half}.");
    }

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::fprop_v2(
                            const tensorflow::OpInputList& row_offsets, 
                            const tensorflow::OpInputList& value_tensors,
                            const tensorflow::Tensor* nnz_array, const std::string& embedding_name, 
                            const bool is_training, tensorflow::Tensor* const forward_result);
template tensorflow::Status EmbeddingWrapper<long long, __half>::fprop_v2(
                            const tensorflow::OpInputList& row_offsets, 
                            const tensorflow::OpInputList& value_tensors,
                            const tensorflow::Tensor* nnz_array, const std::string& embedding_name, 
                            const bool is_training, tensorflow::Tensor* const forward_result);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::fprop_v2(
                            const tensorflow::OpInputList& row_offsets, 
                            const tensorflow::OpInputList& value_tensors,
                            const tensorflow::Tensor* nnz_array, const std::string& embedding_name, 
                            const bool is_training, tensorflow::Tensor* const forward_result);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::fprop_v2(
                            const tensorflow::OpInputList& row_offsets, 
                            const tensorflow::OpInputList& value_tensors,
                            const tensorflow::Tensor* nnz_array, const std::string& embedding_name, 
                            const bool is_training, tensorflow::Tensor* const forward_result);

} // namespace Version1
} // namespace HugeCTR
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

/** This function get input as a single tensor, rather that a list of tensors.
* Its inputs: row_offsets, value_tensors both are single tensors (stack from list of tensor.)
* TODO: Use NCCL to do memory transfer.
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::fprop_v4(
                                const tensorflow::Tensor* row_indices, 
                                const tensorflow::Tensor* values,
                                const std::string& embedding_name,
                                const bool is_training,
                                const cudaStream_t& tf_stream,
                                tensorflow::Tensor* const forward_result) {
    /*get input space*/
    std::string input_space_name = embedding_name;
    input_space_name += (is_training ? "_train" : "_eval");
    std::shared_ptr<InputSpace> space = get_input_space(input_space_name);
    if (!space) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, ", Did not find ", 
                                                    input_space_name, " in input_spaces.");

    /*do distribute keys*/
    const auto distribute_keys_func = get_item_from_map(distribute_keys_on_gpu_func_, embedding_name);
    WRAPPER_REQUIRE_OK((this->*distribute_keys_func)(row_indices, values, embedding_name, is_training, space));


#ifndef NDEBUG
    /*need synchronize streams? wait stream to finish distribute keys on GPU?*/
    HugeCTR::CudaDeviceContext context;
    for (size_t dev_id = 0; dev_id < resource_manager_->get_local_gpu_count(); ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        WRAPPER_CUDA_CHECK(cudaStreamSynchronize(local_gpu->get_stream()));

        // check CSR results
        std::unique_ptr<TypeKey []> host_row_offsets(new TypeKey[space->row_offsets_tensors_[dev_id].get_num_elements()]());
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(host_row_offsets.get(),
                                           space->row_offsets_tensors_[dev_id].get_ptr(),
                                           space->row_offsets_tensors_[dev_id].get_size_in_bytes(),
                                           cudaMemcpyDeviceToHost,
                                           local_gpu->get_stream()));
        std::unique_ptr<TypeKey []> host_values(new TypeKey[space->value_tensors_[dev_id].get_num_elements()]());
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(host_values.get(),
                                           space->value_tensors_[dev_id].get_ptr(),
                                           space->value_tensors_[dev_id].get_size_in_bytes(),
                                           cudaMemcpyDeviceToHost,
                                           local_gpu->get_stream()));
        
        WRAPPER_CUDA_CHECK(cudaStreamSynchronize(local_gpu->get_stream()));

        std::cout << "dev_id = " << dev_id << ", row_offsets = ";
        for (size_t i = 0; i < space->row_offsets_tensors_[dev_id].get_num_elements(); ++i) {
            std::cout << host_row_offsets[i] << ", ";
        } 
        std::cout << std::endl;

        std::cout << "dev_id = " << dev_id << ", values = ";
        for (size_t i = 0; i < space->value_tensors_[dev_id].get_num_elements(); ++i){
            std::cout << host_values[i] << ", ";
        }
        std::cout << std::endl;

    } // for dev_id
#endif


    try {
        /*do forward propagation*/
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

    } catch (const HugeCTR::internal_runtime_error& rt_error) {
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ", rt_error.what());
    }

    /*record cudaEvent on each stream*/
    std::vector<cudaEvent_t> fprop_events = get_item_from_map(fprop_events_, embedding_name);
    if (fprop_events.empty()) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
            "Cannot find fprop cudaEvent_t for embedding: ", embedding_name);
    for (size_t dev_id = 0; dev_id < resource_manager_->get_local_gpu_count(); ++dev_id){
      CudaDeviceContext context;
      const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
      context.set_device(local_gpu->get_device_id());

      WRAPPER_CUDA_CHECK(cudaEventRecord(fprop_events[dev_id], local_gpu->get_stream()));
    }
    /*synchronize tf stream with cuda stream*/
    for (size_t dev_id = 0; dev_id < resource_manager_->get_local_gpu_count(); ++dev_id){
      CudaDeviceContext context;
      const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
      context.set_device(local_gpu->get_device_id());

      WRAPPER_CUDA_CHECK(cudaStreamWaitEvent(tf_stream, fprop_events[dev_id], 0));
    }

    return tensorflow::Status::OK();
}


template tensorflow::Status EmbeddingWrapper<long long, float>::fprop_v4(
                                const tensorflow::Tensor* row_indices, 
                                const tensorflow::Tensor* values,
                                const std::string& embedding_name,
                                const bool is_training,
                                const cudaStream_t& tf_stream,
                                tensorflow::Tensor* const forward_result);
template tensorflow::Status EmbeddingWrapper<long long, __half>::fprop_v4(
                                const tensorflow::Tensor* row_indices, 
                                const tensorflow::Tensor* values,
                                const std::string& embedding_name,
                                const bool is_training,
                                const cudaStream_t& tf_stream,
                                tensorflow::Tensor* const forward_result);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::fprop_v4(
                                const tensorflow::Tensor* row_indices, 
                                const tensorflow::Tensor* values,
                                const std::string& embedding_name,
                                const bool is_training,
                                const cudaStream_t& tf_stream,
                                tensorflow::Tensor* const forward_result);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::fprop_v4(
                                const tensorflow::Tensor* row_indices, 
                                const tensorflow::Tensor* values,
                                const std::string& embedding_name,
                                const bool is_training,
                                const cudaStream_t& tf_stream,
                                tensorflow::Tensor* const forward_result);

} // namespace Version1
} // namespace HugeCTR
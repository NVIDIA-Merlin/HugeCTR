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

/*TODO: IMPORTANT
* This function is used for checking whether distribute keys on GPU can work correctly.
* DO NOT USE IT for other purpose.
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::distribute_keys_gpu(
                                        const tensorflow::Tensor* row_indices,
                                        const tensorflow::Tensor* values,
                                        const std::string& embedding_name, 
                                        const bool is_training, 
                                        tensorflow::Tensor* row_offsets_output,
                                        tensorflow::Tensor* value_tensors_output,
                                        tensorflow::Tensor* nnz_array_output) {
    /*get input space*/
    std::string input_space_name = embedding_name;
    input_space_name += (is_training ? "_train" : "_eval");
    std::shared_ptr<InputSpace> space = get_input_space(input_space_name);
    if (!space) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, ", Did not find ", 
                                                    input_space_name, " in input_spaces.");

    /*do distribute keys*/
    const auto distribute_keys_func = get_item_from_map(distribute_keys_on_gpu_func_, embedding_name);
    WRAPPER_REQUIRE_OK((this->*distribute_keys_func)(row_indices, values, embedding_name, is_training, space));


    /*copy back to tensorflow tensor*/
    auto row_offsets_output_flat = row_offsets_output->flat<long long>();
    auto value_tensors_output_flat = value_tensors_output->flat<TypeKey>();
    auto nnz_array_output_flat = nnz_array_output->flat<long long>();
    HugeCTR::CudaDeviceContext context;
    size_t gpu_count = resource_manager_->get_local_gpu_count();
    std::vector<size_t> host_nnz_array(gpu_count, 0);

    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(row_offsets_output_flat.data() + row_offsets_output->dim_size(1) * dev_id,
                                           space->row_offsets_tensors_[dev_id].get_ptr(),
                                           space->row_offsets_tensors_[dev_id].get_size_in_bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           local_gpu->get_stream()));

        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(value_tensors_output_flat.data() + value_tensors_output->dim_size(1) * dev_id,
                                           space->value_tensors_[dev_id].get_ptr(),
                                           space->value_tensors_[dev_id].get_size_in_bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           local_gpu->get_stream()));
        host_nnz_array[dev_id] = *(space->nnz_array_[dev_id]);
    }
    WRAPPER_CUDA_CHECK(cudaMemcpy(nnz_array_output_flat.data(), 
                                  host_nnz_array.data(),
                                  gpu_count * sizeof(size_t),
                                  cudaMemcpyHostToDevice));

    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        WRAPPER_CUDA_CHECK(cudaStreamSynchronize(local_gpu->get_stream()));
    }
    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::distribute_keys_gpu(
                                        const tensorflow::Tensor* row_indices,
                                        const tensorflow::Tensor* values,
                                        const std::string& embedding_name, 
                                        const bool is_training, 
                                        tensorflow::Tensor* row_offsets_output,
                                        tensorflow::Tensor* value_tensors_output,
                                        tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::distribute_keys_gpu(
                                        const tensorflow::Tensor* row_indices,
                                        const tensorflow::Tensor* values,
                                        const std::string& embedding_name, 
                                        const bool is_training, 
                                        tensorflow::Tensor* row_offsets_output,
                                        tensorflow::Tensor* value_tensors_output,
                                        tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<long long, __half>::distribute_keys_gpu(
                                        const tensorflow::Tensor* row_indices,
                                        const tensorflow::Tensor* values,
                                        const std::string& embedding_name, 
                                        const bool is_training, 
                                        tensorflow::Tensor* row_offsets_output,
                                        tensorflow::Tensor* value_tensors_output,
                                        tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::distribute_keys_gpu(
                                        const tensorflow::Tensor* row_indices,
                                        const tensorflow::Tensor* values,
                                        const std::string& embedding_name, 
                                        const bool is_training, 
                                        tensorflow::Tensor* row_offsets_output,
                                        tensorflow::Tensor* value_tensors_output,
                                        tensorflow::Tensor* nnz_array_output);

} // namespace Version1
} // namespace HugeCTR
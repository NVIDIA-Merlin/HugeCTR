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

/*create an instance of distribute_keys_space*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::DistributeKeysInternelSpaces::create(
                    const std::shared_ptr<HugeCTR::ResourceManager>& resource_manager, 
                    const HugeCTR::Embedding_t& embedding_type,
                    const size_t& batch_size, const size_t& batch_size_eval,
                    const size_t& slot_num, const size_t& max_nnz,
                    std::shared_ptr<DistributeKeysInternelSpaces>& distribute_keys_space) {
    std::shared_ptr<DistributeKeysInternelSpaces> temp_spaces = 
                        std::make_shared<DistributeKeysInternelSpaces>();
    temp_spaces->resource_manager_ = resource_manager;

    /*allocate GeneralBuffer2 and Tensor2 object and vectors*/
    size_t gpu_count = resource_manager->get_local_gpu_count();
    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        temp_spaces->internel_buff_.push_back(HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>::create());
    }
    temp_spaces->binary_flags_.insert(temp_spaces->binary_flags_.begin(), gpu_count, HugeCTR::Tensor2<bool>());
    temp_spaces->cub_coo_indices_output_.insert(temp_spaces->cub_coo_indices_output_.begin(), gpu_count, 
                                                HugeCTR::Tensor2<int>());
    temp_spaces->cub_values_output_.insert(temp_spaces->cub_values_output_.begin(), gpu_count, HugeCTR::Tensor2<TypeKey>());
    temp_spaces->cusparse_csr_row_offsets_output_.insert(temp_spaces->cusparse_csr_row_offsets_output_.begin(), 
                                                         gpu_count, HugeCTR::Tensor2<int>());
    temp_spaces->csr_row_offsets_cast_.insert(temp_spaces->csr_row_offsets_cast_.begin(), gpu_count,
                                              HugeCTR::Tensor2<TypeKey>());
    temp_spaces->copy_input_row_indices_.insert(temp_spaces->copy_input_row_indices_.begin(), gpu_count,
                                                HugeCTR::Tensor2<long long>());
    temp_spaces->copy_input_values_.insert(temp_spaces->copy_input_values_.begin(), gpu_count,
                                           HugeCTR::Tensor2<TypeKey>());
    temp_spaces->cub_temp_storage_bytes_.insert(temp_spaces->cub_temp_storage_bytes_.begin(), gpu_count, 0);
    temp_spaces->cub_d_temp_storage_.insert(temp_spaces->cub_d_temp_storage_.begin(), gpu_count, nullptr);
    temp_spaces->cub_host_num_selected_.insert(temp_spaces->cub_host_num_selected_.begin(), gpu_count, nullptr);
    temp_spaces->cub_dev_num_selected_.insert(temp_spaces->cub_dev_num_selected_.begin(), gpu_count, nullptr);
    temp_spaces->cusparse_handles_.insert(temp_spaces->cusparse_handles_.begin(), gpu_count, nullptr);
    temp_spaces->dev_slot_num_.insert(temp_spaces->dev_slot_num_.begin(), gpu_count, 0);

    /*allocating memory and initialization*/
    const size_t big_batch_size = (batch_size >= batch_size_eval ? batch_size : batch_size_eval);

    HugeCTR::CudaDeviceContext context;
    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        const auto& local_gpu = resource_manager->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        size_t dev_slot_num = 0;
        if (HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash == embedding_type) {
            dev_slot_num = slot_num;
        } else if (HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash == embedding_type ||
                   HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingOneHot == embedding_type) {
            const size_t mod_slots = slot_num % gpu_count;
            dev_slot_num = slot_num / gpu_count + (dev_id < mod_slots ? 1 : 0);
        } else {
            return tensorflow::errors::InvalidArgument(__FILE__, ":", __LINE__, " ",
                                        "Unsupported embedding_type.");
        }
        temp_spaces->dev_slot_num_[dev_id] = dev_slot_num;

        try{
            std::shared_ptr<HugeCTR::BufferBlock2<bool>> bool_block = temp_spaces->internel_buff_[dev_id]->template create_block<bool>();
            std::shared_ptr<HugeCTR::BufferBlock2<TypeKey>> TypeKey_block = temp_spaces->internel_buff_[dev_id]->template create_block<TypeKey>();
            std::shared_ptr<HugeCTR::BufferBlock2<int>> int_block = temp_spaces->internel_buff_[dev_id]->template create_block<int>();
            std::shared_ptr<HugeCTR::BufferBlock2<long long>> longlong_block = temp_spaces->internel_buff_[dev_id]->template create_block<long long>();

            /*register memory space*/
            std::vector<size_t> binary_flags_dims = {1, big_batch_size * slot_num * max_nnz};
            bool_block->reserve(binary_flags_dims, 
                                &(temp_spaces->binary_flags_[dev_id]));

            std::vector<size_t> cub_coo_indices_output_dims = {1, big_batch_size * dev_slot_num * max_nnz};
            int_block->reserve(cub_coo_indices_output_dims,
                                &(temp_spaces->cub_coo_indices_output_[dev_id]));

            std::vector<size_t> cub_values_output_dims = {1, big_batch_size * dev_slot_num * max_nnz};
            TypeKey_block->reserve(cub_values_output_dims,
                                &(temp_spaces->cub_values_output_[dev_id]));

            WRAPPER_CUDA_CHECK(cudaMallocHost(&(temp_spaces->cub_host_num_selected_[dev_id]), sizeof(size_t) * 1));
            WRAPPER_CUDA_CHECK(cudaMalloc(&(temp_spaces->cub_dev_num_selected_[dev_id]), sizeof(size_t) * 1));

            WRAPPER_CUSPARSE_CHECK(cusparseCreate(&(temp_spaces->cusparse_handles_[dev_id])));
            WRAPPER_CUSPARSE_CHECK(cusparseSetStream(temp_spaces->cusparse_handles_[dev_id], local_gpu->get_stream()));

            std::vector<size_t> cusparse_csr_row_offsets_output_dims = {1, big_batch_size * dev_slot_num + 1};
            int_block->reserve(cusparse_csr_row_offsets_output_dims,
                                &(temp_spaces->cusparse_csr_row_offsets_output_[dev_id]));

            std::vector<size_t> csr_row_offsets_cast_dims = {1, big_batch_size * dev_slot_num + 1}; 
            TypeKey_block->reserve(csr_row_offsets_cast_dims,
                                &(temp_spaces->csr_row_offsets_cast_[dev_id]));
                            
            std::vector<size_t> copy_input_row_indices_dims = {1, big_batch_size * slot_num * max_nnz};
            longlong_block->reserve(copy_input_row_indices_dims,
                                &(temp_spaces->copy_input_row_indices_[dev_id]));

            std::vector<size_t> copy_input_values_dims = {1, big_batch_size * slot_num * max_nnz};
            TypeKey_block->reserve(copy_input_values_dims,
                                &(temp_spaces->copy_input_values_[dev_id]));

            /*do allocate GPU memory*/
            temp_spaces->internel_buff_[dev_id]->allocate();
        } catch (const HugeCTR::internal_runtime_error& rt_err){
            return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ", rt_err.what());
        }

    } // for dev_id

    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        const auto& local_gpu = resource_manager->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        size_t cub_temp_storage_bytes_row_indices = 0;
        size_t cub_temp_storage_bytes_values = 0;
        WRAPPER_CUDA_CHECK(CudaUtils::cub_flagged(nullptr, cub_temp_storage_bytes_row_indices,
                                                    temp_spaces->copy_input_row_indices_[dev_id].get_ptr(),
                                                    temp_spaces->binary_flags_[dev_id].get_ptr(),
                                                    temp_spaces->cub_coo_indices_output_[dev_id].get_ptr(),
                                                    temp_spaces->cub_dev_num_selected_[dev_id],
                                                    static_cast<int>(big_batch_size * slot_num * max_nnz),
                                                    local_gpu->get_stream()));

        WRAPPER_CUDA_CHECK(CudaUtils::cub_flagged(nullptr, cub_temp_storage_bytes_values,
                                                    temp_spaces->copy_input_values_[dev_id].get_ptr(),
                                                    temp_spaces->binary_flags_[dev_id].get_ptr(),
                                                    temp_spaces->cub_values_output_[dev_id].get_ptr(),
                                                    temp_spaces->cub_dev_num_selected_[dev_id],
                                                    static_cast<int>(big_batch_size * slot_num * max_nnz),
                                                    local_gpu->get_stream()));
        cub_temp_storage_bytes_row_indices = (cub_temp_storage_bytes_row_indices >= cub_temp_storage_bytes_values 
                                              ? cub_temp_storage_bytes_row_indices
                                              : cub_temp_storage_bytes_values);

        temp_spaces->cub_temp_storage_bytes_[dev_id] = CudaUtils::num_roof(cub_temp_storage_bytes_row_indices, 1ul);
        WRAPPER_CUDA_CHECK(cudaMalloc(&(temp_spaces->cub_d_temp_storage_[dev_id]),
                                        temp_spaces->cub_temp_storage_bytes_[dev_id]));
    } // for dev_id

    distribute_keys_space = temp_spaces;
    return tensorflow::Status::OK();
}



template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::DistributeKeysInternelSpaces::reset() {
    HugeCTR::CudaDeviceContext context;
    for (size_t dev_id = 0; dev_id < resource_manager_->get_local_gpu_count(); ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id()); 
        const auto& stream = local_gpu->get_stream();

        WRAPPER_CUDA_CHECK(cudaMemsetAsync(binary_flags_[dev_id].get_ptr(), 0,
                                           binary_flags_[dev_id].get_size_in_bytes(), stream));
        WRAPPER_CUDA_CHECK(cudaMemsetAsync(cub_coo_indices_output_[dev_id].get_ptr(), 0,
                                           cub_coo_indices_output_[dev_id].get_size_in_bytes(), stream));
        WRAPPER_CUDA_CHECK(cudaMemsetAsync(cub_values_output_[dev_id].get_ptr(), 0,
                                           cub_values_output_[dev_id].get_size_in_bytes(), stream));
        WRAPPER_CUDA_CHECK(cudaMemsetAsync(cusparse_csr_row_offsets_output_[dev_id].get_ptr(), 0,
                                        cusparse_csr_row_offsets_output_[dev_id].get_size_in_bytes(), stream));
        WRAPPER_CUDA_CHECK(cudaMemsetAsync(csr_row_offsets_cast_[dev_id].get_ptr(), 0,
                                           csr_row_offsets_cast_[dev_id].get_size_in_bytes(), stream));
        WRAPPER_CUDA_CHECK(cudaMemsetAsync(copy_input_row_indices_[dev_id].get_ptr(), 0,
                                           copy_input_row_indices_[dev_id].get_size_in_bytes(), stream));
        WRAPPER_CUDA_CHECK(cudaMemsetAsync(copy_input_values_[dev_id].get_ptr(), 0,
                                           copy_input_values_[dev_id].get_size_in_bytes(), stream));
    } // for dev_id

    return tensorflow::Status::OK();
}

template <typename TypeKey, typename TypeFP>
EmbeddingWrapper<TypeKey, TypeFP>::DistributeKeysInternelSpaces::~DistributeKeysInternelSpaces(){
    for (auto& cusparse_handle : cusparse_handles_) {
        if (cusparse_handle) { cusparseDestroy(cusparse_handle); cusparse_handle = nullptr; }
    }
}

template class EmbeddingWrapper<long long, float>::DistributeKeysInternelSpaces;
template class EmbeddingWrapper<long long, __half>::DistributeKeysInternelSpaces;
template class EmbeddingWrapper<unsigned int, float>::DistributeKeysInternelSpaces;
template class EmbeddingWrapper<unsigned int, __half>::DistributeKeysInternelSpaces;

} // namespace Version1
} // namespace HugeCTR
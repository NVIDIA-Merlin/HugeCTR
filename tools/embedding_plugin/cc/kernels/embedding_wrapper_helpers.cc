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

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::try_allocate_distributing_spaces(
        const std::string& space_name, const HugeCTR::Embedding_t& embedding_type,
        const long long& batch_size, const long long& slot_num, const long long& max_nnz) {
    /*check whether already existed*/
    if (distribute_keys_spaces_.find(space_name) != distribute_keys_spaces_.end()) {
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                                           space_name, " distributing spaces have been allocated.",
                                           " Please give an global unique name to allocate new spaces.");
    }

    /*create spaces for distribute keys processing*/
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    std::shared_ptr<DistributeKeysSpaces> temp_distribute_keys_spaces = std::make_shared<DistributeKeysSpaces>(local_gpu_count);
    distribute_keys_spaces_.emplace(std::make_pair(space_name, temp_distribute_keys_spaces));

    // do allocate spaces
    allocate_distribute_keys_spaces_helper(space_name, embedding_type, batch_size, slot_num, max_nnz); 

    switch (embedding_type) {
        case HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash: {
            do_distribute_keys_functor_.reset(new DoDistributedDistributeKeysFunctor());
            break;
        }
        case HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash:
        case HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingOneHot: {
            do_distribute_keys_functor_.reset(new DoLocalizedDistributeKeysFunctor());
            break;
        }
        default: {
            return tensorflow::errors::Unimplemented(__FILE__, ":", __LINE__, " Unsupported embedding type.");
        }
    }

    if (std::is_same<TypeKey, long long>::value) {
        convert_dense_to_csr_functor_.reset(new CudaUtils::ConvertDenseToCSRDoubleFunctor());
    } else if (std::is_same<TypeKey, unsigned int>::value) {
        convert_dense_to_csr_functor_.reset(new CudaUtils::ConvertDenseToCSRFloatFunctor());
    } else {
        return tensorflow::errors::Unimplemented(__FILE__, ":", __LINE__, " Unsupported TypeKey.");
    }


    return tensorflow::Status::OK();
}


template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::allocate_distribute_keys_spaces_helper(
            const std::string& space_name, const HugeCTR::Embedding_t& embedding_type,
            const long long& batch_size, const long long& slot_num, const long long& max_nnz) {
    /*get distribute keys spaces*/
    auto distribute_keys_spaces = get_distribute_keys_spaces(space_name);
    if (!distribute_keys_spaces) return tensorflow::errors::NotFound(__FILE__, ":", __LINE__, " ",
                                        "Not found spaces for distributing keys of ", space_name);
    if (distribute_keys_spaces->allocated_) tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " "
                                        "distribute_keys_spaces of ", space_name, " are already allocated.");

    /*calculate some dimentions*/
    size_t num_row_offset = batch_size * slot_num + 1;
    size_t num_max_value = batch_size * slot_num * max_nnz;
    if (num_row_offset <= 0 || num_max_value <=0) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                                        "The dimension of row_offset or value_tensor is zero. Please check Attr:",
                                        "batch_size or slot_num or max_nnz."); 
    
    // CudaDeviceContext context;
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id) {
        // int cur_device = resource_manager_->get_local_gpu(dev_id)->get_device_id();
        // context.set_device(cur_device);
        cudaStream_t temp_stream;
        WRAPPER_CUDA_CHECK(cudaStreamCreate(&temp_stream));
        distribute_keys_spaces->cuda_streams_.emplace_back(temp_stream);

        // cusparseHandle_t temp_cusparse_handle;
        // WRAPPER_CUSPARSE_CHECK(cusparseCreate(&temp_cusparse_handle));
        // WRAPPER_CUSPARSE_CHECK(cusparseSetStream(temp_cusparse_handle,
        //                                         //  resource_manager_->get_local_gpu(dev_id)->get_stream()
        //                                         distribute_keys_spaces->cuda_streams_[dev_id]
        //                                          ));
        // distribute_keys_spaces->cusparse_handles_.emplace_back(temp_cusparse_handle);

        // cublasHandle_t temp_cublas_handle;
        // WRAPPER_CUBLAS_CHECK(cublasCreate(&temp_cublas_handle));
        // WRAPPER_CUBLAS_CHECK(cublasSetStream(temp_cublas_handle, 
        //                                     //  resource_manager_->get_local_gpu(dev_id)->get_stream()
        //                                     distribute_keys_spaces->cuda_streams_[dev_id]
        //                                      ));
        // distribute_keys_spaces->cublas_handles_.emplace_back(temp_cublas_handle);

        WRAPPER_CUDA_CHECK(cudaMalloc(&(distribute_keys_spaces->input_keys_copies_[dev_id]),
                                      sizeof(TypeKey) * num_max_value));

        WRAPPER_CUDA_CHECK(cudaMalloc(&(distribute_keys_spaces->csr_values_[dev_id]), 
                                       sizeof(TypeKey) * num_max_value));
        WRAPPER_CUDA_CHECK(cudaMalloc(&(distribute_keys_spaces->csr_row_offsets_[dev_id]), 
                                        sizeof(int) * num_row_offset));
        WRAPPER_CUDA_CHECK(cudaMalloc(&(distribute_keys_spaces->csr_row_offsets_casts_[dev_id]),
                                        sizeof(long long) * num_row_offset));
        WRAPPER_CUDA_CHECK(cudaMalloc(&(distribute_keys_spaces->csr_col_indices_[dev_id]),
                                        sizeof(int) * num_max_value));

        WRAPPER_CUDA_CHECK(cudaMalloc(&(distribute_keys_spaces->csr_nnz_rows_[dev_id]),
                                        sizeof(int) * (num_row_offset - 1)));
        WRAPPER_CUDA_CHECK(cudaMalloc(&(distribute_keys_spaces->input_keys_transposes_[dev_id]),
                                        sizeof(TypeKey) * num_max_value));

        cusparseMatDescr_t temp_desc;
        WRAPPER_CUSPARSE_CHECK(cusparseCreateMatDescr(&temp_desc));
        distribute_keys_spaces->cusparse_mat_descs_.emplace_back(temp_desc);
    }

    cusparseHandle_t temp_cusparse_handle;
    WRAPPER_CUSPARSE_CHECK(cusparseCreate(&temp_cusparse_handle));
    distribute_keys_spaces->cusparse_handles_.emplace_back(temp_cusparse_handle);

    cublasHandle_t temp_cublas_handle;
    WRAPPER_CUBLAS_CHECK(cublasCreate(&temp_cublas_handle));
    distribute_keys_spaces->cublas_handles_.emplace_back(temp_cublas_handle);


    distribute_keys_spaces->batch_size_ = batch_size;
    distribute_keys_spaces->slot_num_ = slot_num;
    distribute_keys_spaces->max_nnz_ = max_nnz;
    distribute_keys_spaces->allocated_ = true;
    distribute_keys_spaces->embedding_type_ = embedding_type;
    return tensorflow::Status::OK();
}

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::distributed_embedding_distribute_keys_helper(
                                                        std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                                                        const tensorflow::Tensor* input_keys,
                                                        std::vector<tensorflow::Tensor*>& row_offset_output,
                                                        std::vector<tensorflow::Tensor*>& value_tensor_output,
                                                        tensorflow::Tensor* nnz_array_output) {
    if (distribute_keys_space->allocated_ == false) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                                                        "distribute_keys_spaces have not been allocated.");

    // CudaDeviceContext context;
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();

    auto input_keys_flat = input_keys->flat<TypeKey>();
    auto nnz_array_output_flat = nnz_array_output->flat<long long>();
    
    for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id){
        // int cur_device = resource_manager_->get_local_gpu(dev_id)->get_device_id();
        // context.set_device(cur_device);

        /*copy input_keys to each GPU spaces*/
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(distribute_keys_space->input_keys_copies_[dev_id], 
                                           input_keys_flat.data(),
                                           sizeof(TypeKey) * input_keys_flat.size(),
                                           cudaMemcpyHostToDevice,
                                        //    resource_manager_->get_local_gpu(dev_id)->get_stream()
                                           distribute_keys_space->cuda_streams_[dev_id]
                                           ));
    }

    for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id){
        // int cur_device = resource_manager_->get_local_gpu(dev_id)->get_device_id();
        // context.set_device(cur_device);

        /*all value + 1, to erase -1 which is used as invalid keys*/
        CudaUtils::all_keys_plus_1(distribute_keys_space->input_keys_copies_[dev_id], 
                                   input_keys_flat.size(),
                                   resource_manager_->get_local_gpu(dev_id)->get_sm_count(),
                                //    resource_manager_->get_local_gpu(dev_id)->get_stream()
                                   distribute_keys_space->cuda_streams_[dev_id]
                                   );

        /*erase each copies' keys which donot belong to that GPU*/
        CudaUtils::erase_distributed_embedding_keys(distribute_keys_space->input_keys_copies_[dev_id],
                                                    dev_id, local_gpu_count, input_keys_flat.size(),
                                                    resource_manager_->get_local_gpu(dev_id)->get_sm_count(),
                                                    // resource_manager_->get_local_gpu(dev_id)->get_stream()
                                                    distribute_keys_space->cuda_streams_[dev_id]
                                                    );

        // /* fused all value + 1 and erase -1 which is used as invalid keys*/
        // CudaUtils::fuse_keys_plus_erase_distributed_embedding(distribute_keys_space->input_keys_copies_[dev_id],
        //                                                       dev_id, local_gpu_count, input_keys_flat.size(),
        //                                                       resource_manager_->get_local_gpu(dev_id)->get_sm_count(),
        //                                                     //   resource_manager_->get_local_gpu(dev_id)->get_stream()
        //                                                       distribute_keys_space->cuda_streams_[dev_id]
        //                                                       );

        /*convert to CSR format*/
        WRAPPER_CUSPARSE_CHECK(cusparseSetStream(distribute_keys_space->cusparse_handles_[0],
                                                 distribute_keys_space->cuda_streams_[dev_id]));
        WRAPPER_CUBLAS_CHECK(cublasSetStream(distribute_keys_space->cublas_handles_[0],
                                             distribute_keys_space->cuda_streams_[dev_id]));
        // CudaUtils::convert_dense_to_csr(distribute_keys_space->input_keys_copies_[dev_id],
        //                                 distribute_keys_space->batch_size_ * distribute_keys_space->slot_num_, 
        //                                 distribute_keys_space->max_nnz_,
        //                                 distribute_keys_space->cusparse_handles_[0],
        //                                 distribute_keys_space->cublas_handles_[0],
        //                                 distribute_keys_space->csr_values_[dev_id],
        //                                 distribute_keys_space->csr_row_offsets_[dev_id],
        //                                 distribute_keys_space->csr_col_indices_[dev_id],
        //                                 &(distribute_keys_space->total_nnzs_[dev_id]),
        //                                 distribute_keys_space->cusparse_mat_descs_[dev_id],
        //                                 distribute_keys_space->csr_nnz_rows_[dev_id],
        //                                 distribute_keys_space->input_keys_transposes_[dev_id]);
        (*convert_dense_to_csr_functor_)(distribute_keys_space->input_keys_copies_[dev_id],
                                         distribute_keys_space->batch_size_ * distribute_keys_space->slot_num_, 
                                         distribute_keys_space->max_nnz_,
                                         distribute_keys_space->cusparse_handles_[0],
                                         distribute_keys_space->cublas_handles_[0],
                                         distribute_keys_space->csr_values_[dev_id],
                                         distribute_keys_space->csr_row_offsets_[dev_id],
                                         distribute_keys_space->csr_col_indices_[dev_id],
                                         &(distribute_keys_space->total_nnzs_[dev_id]),
                                         distribute_keys_space->cusparse_mat_descs_[dev_id],
                                         distribute_keys_space->csr_nnz_rows_[dev_id],
                                         distribute_keys_space->input_keys_transposes_[dev_id]);

        /*csr value tensor -1, because +1 was carried out.*/
        CudaUtils::value_tensors_subtract_1(distribute_keys_space->csr_values_[dev_id], 
                    distribute_keys_space->batch_size_ * distribute_keys_space->slot_num_ * distribute_keys_space->max_nnz_,
                                            resource_manager_->get_local_gpu(dev_id)->get_sm_count(),
                                            // resource_manager_->get_local_gpu(dev_id)->get_stream()
                                            distribute_keys_space->cuda_streams_[dev_id]
                                            );

        /*cast csr row offset to output type*/
        CudaUtils::cast_elements(distribute_keys_space->csr_row_offsets_[dev_id],
                                 distribute_keys_space->csr_row_offsets_casts_[dev_id],
                                 distribute_keys_space->batch_size_ * distribute_keys_space->slot_num_ + 1,
                                 resource_manager_->get_local_gpu(dev_id)->get_sm_count(),
                                //  resource_manager_->get_local_gpu(dev_id)->get_stream()
                                 distribute_keys_space->cuda_streams_[dev_id]
                                 );
    }

    for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id){
        // int cur_device = resource_manager_->get_local_gpu(dev_id)->get_device_id();
        // context.set_device(cur_device);

        /*copy each CSR format results to the output tensor*/
        auto row_offset_output_flat = row_offset_output[dev_id]->flat<long long>();
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(row_offset_output_flat.data(),
                                           distribute_keys_space->csr_row_offsets_casts_[dev_id],
            sizeof(long long) * (distribute_keys_space->batch_size_ * distribute_keys_space->slot_num_ + 1),
                                           cudaMemcpyDeviceToHost,
                                        //    resource_manager_->get_local_gpu(dev_id)->get_stream()
                                            distribute_keys_space->cuda_streams_[dev_id]
                                           ));
        auto value_tensor_output_flat = value_tensor_output[dev_id]->flat<TypeKey>();
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(value_tensor_output_flat.data(),
                                           distribute_keys_space->csr_values_[dev_id],
            sizeof(TypeKey) * (distribute_keys_space->batch_size_ * distribute_keys_space->slot_num_ * distribute_keys_space->max_nnz_),
                                           cudaMemcpyDeviceToHost,
                                        //    resource_manager_->get_local_gpu(dev_id)->get_stream()
                                           distribute_keys_space->cuda_streams_[dev_id]
                                           ));
        nnz_array_output_flat(dev_id) = distribute_keys_space->total_nnzs_[dev_id];
    }

    for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id){
        // int cur_device = resource_manager_->get_local_gpu(dev_id)->get_device_id();
        // context.set_device(cur_device);
        // WRAPPER_CUDA_CHECK(cudaStreamSynchronize(resource_manager_->get_local_gpu(dev_id)->get_stream()));
        WRAPPER_CUDA_CHECK(cudaStreamSynchronize(distribute_keys_space->cuda_streams_[dev_id]));
    }
    
    return tensorflow::Status::OK();
}



/** This function is used to distribute keys to each GPU CSR format through wrapper.
* 
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::do_distributing_keys(
            const std::string& space_name, const tensorflow::Tensor* input_keys,
            std::vector<tensorflow::Tensor*>& row_offset_output,
            std::vector<tensorflow::Tensor*>& value_tensor_output,
            tensorflow::Tensor* nnz_array_output) {
    auto distribute_keys_spaces = get_distribute_keys_spaces(space_name);
    if (!distribute_keys_spaces) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                                        "distribute_keys_spaces with name: ", space_name, " does not exist.");
    if (false == distribute_keys_spaces->allocated_) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                                        "distribute_keys_spaces have not allocated GPU memory.");

    if (distribute_keys_spaces->gpu_count_ != row_offset_output.size()) 
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                "Attr: gpu_count in hugectr.distribute_keys() should be equal to the number which is initialized",
                "in hugectr.init()."); 

    // tensorflow::Status status;
    // switch(distribute_keys_spaces->embedding_type_) {
    //     case HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash: {
    //         status =  distributed_embedding_distribute_keys_helper(distribute_keys_spaces,
    //                                                                input_keys,
    //                                                                row_offset_output,
    //                                                                value_tensor_output,
    //                                                                nnz_array_output);
    //         break;
    //     }
    //     case HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingOneHot:
    //     case HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash: {
    //         // status =  localized_embedding_distribute_keys_helper();
    //         return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
    //                                             "Not implemented.");
    //         break;
    //     }
    //     default: {
    //         return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
    //                                           "No such embedding_type");
    //     }
    // }

    tensorflow::Status status = (*do_distribute_keys_functor_)(this, 
                                                               distribute_keys_spaces,
                                                               input_keys,
                                                               row_offset_output,
                                                               value_tensor_output,
                                                               nnz_array_output);

    return status;
}

template tensorflow::Status EmbeddingWrapper<long long, float>::try_allocate_distributing_spaces(
        const std::string& space_name, const HugeCTR::Embedding_t& embedding_type,
        const long long& batch_size, const long long& slot_num, const long long& max_nnz);
template tensorflow::Status EmbeddingWrapper<long long, __half>::try_allocate_distributing_spaces(
        const std::string& space_name, const HugeCTR::Embedding_t& embedding_type,
        const long long& batch_size, const long long& slot_num, const long long& max_nnz);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::try_allocate_distributing_spaces(
        const std::string& space_name, const HugeCTR::Embedding_t& embedding_type,
        const long long& batch_size, const long long& slot_num, const long long& max_nnz);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::try_allocate_distributing_spaces(
        const std::string& space_name, const HugeCTR::Embedding_t& embedding_type,
        const long long& batch_size, const long long& slot_num, const long long& max_nnz);

template tensorflow::Status EmbeddingWrapper<long long, float>::allocate_distribute_keys_spaces_helper(
            const std::string& space_name, const HugeCTR::Embedding_t& embedding_type,
            const long long& batch_size, const long long& slot_num, const long long& max_nnz);
template tensorflow::Status EmbeddingWrapper<long long, __half>::allocate_distribute_keys_spaces_helper(
            const std::string& space_name, const HugeCTR::Embedding_t& embedding_type,
            const long long& batch_size, const long long& slot_num, const long long& max_nnz);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::allocate_distribute_keys_spaces_helper(
            const std::string& space_name, const HugeCTR::Embedding_t& embedding_type,
            const long long& batch_size, const long long& slot_num, const long long& max_nnz);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::allocate_distribute_keys_spaces_helper(
            const std::string& space_name, const HugeCTR::Embedding_t& embedding_type,
            const long long& batch_size, const long long& slot_num, const long long& max_nnz);

template tensorflow::Status EmbeddingWrapper<long long, float>::distributed_embedding_distribute_keys_helper(
            std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
            const tensorflow::Tensor* input_keys,
            std::vector<tensorflow::Tensor*>& row_offset_output,
            std::vector<tensorflow::Tensor*>& value_tensor_output,
            tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<long long, __half>::distributed_embedding_distribute_keys_helper(
            std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
            const tensorflow::Tensor* input_keys,
            std::vector<tensorflow::Tensor*>& row_offset_output,
            std::vector<tensorflow::Tensor*>& value_tensor_output,
            tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::distributed_embedding_distribute_keys_helper(
            std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
            const tensorflow::Tensor* input_keys,
            std::vector<tensorflow::Tensor*>& row_offset_output,
            std::vector<tensorflow::Tensor*>& value_tensor_output,
            tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::distributed_embedding_distribute_keys_helper(
            std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
            const tensorflow::Tensor* input_keys,
            std::vector<tensorflow::Tensor*>& row_offset_output,
            std::vector<tensorflow::Tensor*>& value_tensor_output,
            tensorflow::Tensor* nnz_array_output);

template tensorflow::Status EmbeddingWrapper<long long, float>::do_distributing_keys(
            const std::string& space_name, const tensorflow::Tensor* input_keys,
            std::vector<tensorflow::Tensor*>& row_offset_output,
            std::vector<tensorflow::Tensor*>& value_tensor_output,
            tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<long long, __half>::do_distributing_keys(
            const std::string& space_name, const tensorflow::Tensor* input_keys,
            std::vector<tensorflow::Tensor*>& row_offset_output,
            std::vector<tensorflow::Tensor*>& value_tensor_output,
            tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::do_distributing_keys(
            const std::string& space_name, const tensorflow::Tensor* input_keys,
            std::vector<tensorflow::Tensor*>& row_offset_output,
            std::vector<tensorflow::Tensor*>& value_tensor_output,
            tensorflow::Tensor* nnz_array_output);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::do_distributing_keys(
            const std::string& space_name, const tensorflow::Tensor* input_keys,
            std::vector<tensorflow::Tensor*>& row_offset_output,
            std::vector<tensorflow::Tensor*>& value_tensor_output,
            tensorflow::Tensor* nnz_array_output);
            
} // namespace Version1
} // namespace HugeCTR
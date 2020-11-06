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

#include "wrapper_variables.h"
#include "embedding_utils.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "cuda_utils.h"
#include <memory>
#include <type_traits>

#include <iostream>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 



/* This op is like distribute_keys_v2. But this one will create cudaStreams on each 
forward process (Compute function.)
TODO: need more modification. Just like distribute_keys_v2.
*/
template <typename Device>
class EmbeddingDistributeKeysV4Op : public OpKernel {
public:
    explicit EmbeddingDistributeKeysV4Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
        // FIXME: Therefore, this op should not be called.
        OP_REQUIRES(ctx, false, errors::Unavailable(__FILE__, ":", __LINE__, " Should not use this op."));

        OP_REQUIRES_OK(ctx, ctx->GetAttr("gpu_count", &gpu_count_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_type", &embedding_type_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_feature_num", &max_feature_num_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_nnz", &max_nnz_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_num", &slot_num_));

        /*do sth in advance.*/
        num_row_offset_dim_ = batch_size_ * slot_num_ + 1;
        num_max_value_dim_ = (max_nnz_ * slot_num_ <= max_feature_num_)
                             ? max_nnz_ * slot_num_ * batch_size_
                             : max_feature_num_ * batch_size_;

        // cudaSetDevice(0); // set gpu
        /*allocate spaces*/
        // all keys copies
        keys_ptr_.insert(keys_ptr_.begin(), gpu_count_, nullptr); 
        // csr buffers
        csr_values_.insert(csr_values_.begin(), gpu_count_, nullptr);
        csr_row_offset_.insert(csr_row_offset_.begin(), gpu_count_, nullptr);
        csr_col_indices_.insert(csr_col_indices_.begin(), gpu_count_, nullptr);
        nnz_array_.insert(nnz_array_.begin(), gpu_count_, 0);
        // output tensors
        row_offsets_output_.insert(row_offsets_output_.begin(), gpu_count_, nullptr);
        value_tensors_output_.insert(value_tensors_output_.begin(), gpu_count_, nullptr);
        // interner spaces  used in convering process
        nnz_row_.insert(nnz_row_.begin(), gpu_count_, nullptr);
        keys_ptr_transpose_.insert(keys_ptr_transpose_.begin(), gpu_count_, nullptr);
        // only used in localized embedding
        if ("localized" == embedding_type_) {
            csr_row_offset_binary_flag_.insert(csr_row_offset_binary_flag_.begin(), gpu_count_, nullptr);
            csr_row_offset_interner_output_.insert(csr_row_offset_interner_output_.begin(), gpu_count_, nullptr);
            csr_row_offset_temp_storage_.insert(csr_row_offset_temp_storage_.begin(), gpu_count_, nullptr);
            csr_row_offset_temp_storage_bytes_.insert(csr_row_offset_temp_storage_bytes_.begin(), gpu_count_, 0);
            d_num_selected_out_.insert(d_num_selected_out_.begin(), gpu_count_, nullptr);
        }

        int original_device = 0;
        PLUGIN_CUDA_CHECK(ctx, cudaGetDevice(&original_device));
        PLUGIN_CUDA_CHECK(ctx, cudaSetDevice(1));
        for (int i = 0; i < gpu_count_; ++i) {
            PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&keys_ptr_[i], sizeof(long long) * batch_size_ * slot_num_ * max_nnz_));
            PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&csr_values_[i], sizeof(long long) * num_max_value_dim_));
            PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&csr_row_offset_[i], sizeof(int) * num_row_offset_dim_));
            PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&csr_col_indices_[i], sizeof(int) * num_max_value_dim_));

            PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&nnz_row_[i], sizeof(int) * batch_size_ * slot_num_));
            PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&keys_ptr_transpose_[i], sizeof(long long) * batch_size_ * slot_num_ * max_nnz_));

            if ("localized" == embedding_type_) {
                PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&csr_row_offset_binary_flag_[i], sizeof(int) * num_row_offset_dim_));
                PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&csr_row_offset_interner_output_[i], sizeof(int) * num_row_offset_dim_));
                PLUGIN_CUDA_CHECK(ctx, CudaUtils::get_temp_storage_bytes(csr_row_offset_[i], csr_row_offset_binary_flag_[i],
                                                                         csr_row_offset_interner_output_[i], 
                                                                         num_row_offset_dim_, 
                                                                         csr_row_offset_temp_storage_bytes_[i]));
                csr_row_offset_temp_storage_bytes_[i] = CudaUtils::num_roof(csr_row_offset_temp_storage_bytes_[i],
                                                                                         sizeof(int));
                std::cout << "Allocating " << csr_row_offset_temp_storage_bytes_[i] << " bytes for " 
                          << i << "'s temp storage." << std::endl;
                PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&csr_row_offset_temp_storage_[i], csr_row_offset_temp_storage_bytes_[i]));
                PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&d_num_selected_out_[i], sizeof(int) * 1));
            }
        }

        temp_csr_row_offset_.insert(temp_csr_row_offset_.begin(), 1, nullptr);
        PLUGIN_CUDA_CHECK(ctx, cudaMalloc(&temp_csr_row_offset_[0], sizeof(long long) * num_row_offset_dim_));

        // create handle
        PLUGIN_CUSPARSE_CHECK(ctx, cusparseCreate(&cusparse_handle_));
        PLUGIN_CUBLAS_CHECK(ctx, cublasCreate(&cublas_handle_));

        // MatDescriptor
        cusparseCreateMatDescr(&desc_);

        if ("distributed" == embedding_type_) {
            erase_embedding_keys_functor_.reset(new CudaUtils::EraseDistributedEmbeddingKeysFunctor<long long>());
        } else if ("localized" == embedding_type_) {
            erase_embedding_keys_functor_.reset(new CudaUtils::EraseLocalizedEmbeddingKeysFunctor<long long>());
        }
        convert_dense_to_csr_functor_.reset(new CudaUtils::ConvertDenseToCSRDoubleFunctor());

        PLUGIN_CUDA_CHECK(ctx, cudaSetDevice(original_device));
    }

    ~EmbeddingDistributeKeysV4Op() {
        if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
        if (cublas_handle_) cublasDestroy(cublas_handle_);
        // for (auto stream: cuda_streams_) {
        //     if (stream) cudaStreamDestroy(stream);
        // }
        if (desc_) cusparseDestroyMatDescr(desc_);
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* all_keys = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("all_keys", &all_keys)); // keys in dense tensor
        PLUGIN_RETURN_IF_TRUE(ctx, all_keys->dims() != 3, 
                            "all_keys is a dense tensor, and its shape should be [batch_size, slot_num, max_nnz].");
        PLUGIN_RETURN_IF_TRUE(ctx, all_keys->dim_size(0) != batch_size_, "all_keys's batch_size is not equal to this op's attr.");
        PLUGIN_RETURN_IF_TRUE(ctx, all_keys->dim_size(1) != slot_num_, "all_keys's slot_num is not equal to this op's attr.");
        PLUGIN_RETURN_IF_TRUE(ctx, all_keys->dim_size(2) != max_nnz_, "all_keys's max_nnz is not equal to this op's attr.");

        // const GPUDevice& device = ctx->eigen_gpu_device();
        // cudaStream_t stream = device.stream();
        // cusparseSetStream(cusparse_handle_, stream);
        // cublasSetStream(cublas_handle_, stream);

        int original_device = 0;
        PLUGIN_CUDA_CHECK(ctx, cudaGetDevice(&original_device));
        // PLUGIN_CUDA_CHECK(ctx, cudaDeviceEnablePeerAccess(1, 0));

        PLUGIN_CUDA_CHECK(ctx, cudaSetDevice(1));
        // PLUGIN_CUDA_CHECK(ctx, cudaDeviceEnablePeerAccess(original_device, 0));
        cuda_streams_.clear();
        for (int i = 0; i < gpu_count_; ++i) {
            cudaStream_t temp_stream;
            cudaStreamCreate(&temp_stream);
            cuda_streams_.emplace_back(temp_stream);
        }

        /*copy all keys for gpu_count times*/
        auto all_keys_flat = all_keys->flat<long long>();
        for (int i = 0; i < gpu_count_; ++i) {
            PLUGIN_CUDA_CHECK(ctx, cudaMemcpyAsync(keys_ptr_[i], all_keys_flat.data(), 
                                    sizeof(long long) * all_keys_flat.size(),
                                    cudaMemcpyHostToDevice, cuda_streams_[i]));
            if ("localized" == embedding_type_) {
                CudaUtils::generate_binary_vec(csr_row_offset_binary_flag_[i], num_row_offset_dim_, i, 
                                               slot_num_, gpu_count_, 32, cuda_streams_[i]);
            }
        }
        
        /*each key + 1, and modify the keys which do not belong to that GPU to 0*/
        // each key + 1
        for (int i = 0; i < gpu_count_; ++i) {
            CudaUtils::all_keys_plus_1(keys_ptr_[i], all_keys_flat.size(), 32, cuda_streams_[i]);
        }
        
        // erase keys
        for (int i = 0; i < gpu_count_; ++i) {
            CudaUtils::erase_distributed_embedding_keys(keys_ptr_[i], i, gpu_count_, all_keys_flat.size(), 32, cuda_streams_[i]);
            // (*erase_embedding_keys_functor_)(keys_ptr_[i], i, all_keys->dim_size(1), all_keys->dim_size(2), gpu_count_, 
                                            // all_keys_flat.size(), 32, cuda_streams_[i]);

        }
        
        /*convert each dense tensor to CSR*/
        for (int i = 0; i < gpu_count_; ++i) {
            PLUGIN_CUSPARSE_CHECK(ctx, cusparseSetStream(cusparse_handle_, cuda_streams_[i]));
            PLUGIN_CUBLAS_CHECK(ctx, cublasSetStream(cublas_handle_, cuda_streams_[i]));

            CudaUtils::convert_dense_to_csr(keys_ptr_[i], batch_size_ * slot_num_, max_nnz_, cusparse_handle_,
                                            cublas_handle_,
                                            csr_values_[i], csr_row_offset_[i], csr_col_indices_[i],
                                            &nnz_array_[i], desc_, nnz_row_[i], keys_ptr_transpose_[i]);
            // (*convert_dense_to_csr_functor_)(keys_ptr_[i], batch_size_ * slot_num_, max_nnz_, cusparse_handle_,
            //                                 cublas_handle_,
            //                                 csr_values_[i], csr_row_offset_[i], csr_col_indices_[i],
            //                                 &nnz_array_[i], desc_, nnz_row_[i], keys_ptr_transpose_[i]);


            CudaUtils::value_tensors_subtract_1(csr_values_[i], num_max_value_dim_, 32, cuda_streams_[i]); // each key - 1
            if ("localized" == embedding_type_) { // select slot
                PLUGIN_CUDA_CHECK(ctx, CudaUtils::select_slots(csr_row_offset_[i], csr_row_offset_interner_output_[i], 
                                                               num_row_offset_dim_, csr_row_offset_temp_storage_[i],
                                                               csr_row_offset_binary_flag_[i],
                                                               csr_row_offset_temp_storage_bytes_[i],
                                                               d_num_selected_out_[i],
                                                               cuda_streams_[i]));
            }

        }

        PLUGIN_CUDA_CHECK(ctx, cudaSetDevice(original_device));
        /*allocate output*/
        for (int i = 0; i < gpu_count_; ++i) {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {num_row_offset_dim_}, &row_offsets_output_[i]));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(gpu_count_ + i, {num_max_value_dim_}, &value_tensors_output_[i]));
        }
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2 * gpu_count_, {gpu_count_}, &nnz_array_output_));


        /*copy to output*/
        for (int i = 0; i < gpu_count_; ++i) {
            PLUGIN_CUDA_CHECK(ctx, cudaSetDevice(1));
            if ("distributed" == embedding_type_) {
                CudaUtils::cast_elements(csr_row_offset_[i], temp_csr_row_offset_[0], num_row_offset_dim_, 32, cuda_streams_[i]);
            } else {
                CudaUtils::cast_elements(csr_row_offset_interner_output_[i], temp_csr_row_offset_[0], 
                                         num_row_offset_dim_, 32, cuda_streams_[i]);
            }

            PLUGIN_CUDA_CHECK(ctx, cudaSetDevice(original_device));
            auto row_offset_output_flat = row_offsets_output_[i]->flat<long long>();
            PLUGIN_CUDA_CHECK(ctx, cudaMemcpyAsync(row_offset_output_flat.data(), temp_csr_row_offset_[0], 
                                    sizeof(long long) * row_offset_output_flat.size(),
                                    cudaMemcpyDeviceToHost, cuda_streams_[i]));

            auto value_tensors_output_flat = value_tensors_output_[i]->flat<long long>();
            PLUGIN_CUDA_CHECK(ctx, cudaMemcpyAsync(value_tensors_output_flat.data(), csr_values_[i],
                                    sizeof(long long)* value_tensors_output_flat.size(),
                                    cudaMemcpyDeviceToHost, cuda_streams_[i]));
        }
        auto nnz_array_output_flat = nnz_array_output_->flat<long long>();

        for (int i = 0; i < gpu_count_; ++i) { PLUGIN_CUDA_CHECK(ctx, cudaStreamSynchronize(cuda_streams_[i])); }

        PLUGIN_CUDA_CHECK(ctx, cudaMemcpyAsync(nnz_array_output_flat.data(), nnz_array_.data(), 
                        sizeof(long long) * gpu_count_, cudaMemcpyHostToHost,
                        cuda_streams_[0]));
        for (int i = 0; i < gpu_count_; ++i) { PLUGIN_CUDA_CHECK(ctx, cudaStreamSynchronize(cuda_streams_[i])); }


        for (auto stream: cuda_streams_) {
            if (stream) cudaStreamDestroy(stream);
        }

        
    }
private:
    int gpu_count_;
    std::string embedding_type_;
    int max_feature_num_;
    int max_nnz_;
    int batch_size_;
    int slot_num_;
    std::vector<long long*, CudaUtils::CudaAllocator<long long*>> keys_ptr_;
    cusparseHandle_t cusparse_handle_;
    cublasHandle_t cublas_handle_;
    std::vector<long long*, CudaUtils::CudaAllocator<long long*>> csr_values_;
    std::vector<int*, CudaUtils::CudaAllocator<int*>> csr_row_offset_;
    std::vector<int*, CudaUtils::CudaAllocator<int*>> csr_col_indices_;
    long long num_row_offset_dim_;
    long long num_max_value_dim_;
    std::vector<long long> nnz_array_;
    std::vector<Tensor*> row_offsets_output_;
    std::vector<Tensor*> value_tensors_output_;
    Tensor* nnz_array_output_ = nullptr;
    std::vector<long long*, CudaUtils::CudaAllocator<long long*>> temp_csr_row_offset_;
    std::vector<cudaStream_t> cuda_streams_;
    cusparseMatDescr_t desc_;
    std::vector<int*, CudaUtils::CudaAllocator<int*>> nnz_row_;
    std::vector<long long*, CudaUtils::CudaAllocator<long long*>> keys_ptr_transpose_;
    // only used in localized embedding
    std::vector<int*, CudaUtils::CudaAllocator<int*>> csr_row_offset_binary_flag_;
    std::vector<int*, CudaUtils::CudaAllocator<int*>> csr_row_offset_interner_output_;
    std::vector<int*, CudaUtils::CudaAllocator<int*>> csr_row_offset_temp_storage_;
    std::vector<size_t> csr_row_offset_temp_storage_bytes_;
    std::vector<int*, CudaUtils::CudaAllocator<int*>> d_num_selected_out_;
    std::unique_ptr<CudaUtils::EraseEmbeddingKeysFunctor> erase_embedding_keys_functor_;
    std::unique_ptr<CudaUtils::ConvertDenseToCSRFunctor> convert_dense_to_csr_functor_;
};

REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingDistributeKeysV4").Device(DEVICE_CPU), 
                        EmbeddingDistributeKeysV4Op<CPUDevice>);



} // namespace tensorflow
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
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "cuda_runtime.h"
#include "cusparse.h"
#include "cublas_v2.h"
#include <string>
#include <type_traits>

namespace CudaUtils {

#define PLUGIN_CUDA_CHECK(ctx, cmd)                                                 \
    do {                                                                            \
        cudaError_t error = (cmd);                                                  \
        if (error != cudaSuccess) {                                                 \
            ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ", \
                            cudaGetErrorString(error)));                            \
            return;                                                                 \
        }                                                                           \
    } while (0)

#define PLUGIN_CUDA_CHECK_ASYNC(ctx, cmd, callback)                                     \
    do {                                                                                \
        cudaError_t error = (cmd);                                                      \
        if (error != cudaSuccess) {                                                     \
            (ctx)->CtxFailure(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ", \
                              cudaGetErrorString(error)));                              \
            (callback)();                                                               \
            return;                                                                     \
        }                                                                               \
    } while (0)    

#define WRAPPER_CUDA_CHECK(cmd)                                                 \
    do {                                                                        \
        cudaError_t error = (cmd);                                              \
        if (error != cudaSuccess) {                                             \
            return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",    \
                            cudaGetErrorString(error));                         \
        }                                                                       \
    } while(0)

#define PLUGIN_RETURN_IF_TRUE(ctx, condition, message)                              \
    do {                                                                             \
        if ((condition)) {                                                          \
            ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ", \
                            (message)));                                             \
            return;                                                                  \
        }                                                                            \
    } while (0) 

#define PLUGIN_CUSPARSE_CHECK(ctx, cmd)                                             \
    do {                                                                            \
        cusparseStatus_t error = (cmd);                                             \
        if (error != CUSPARSE_STATUS_SUCCESS) {                                     \
            ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ", \
                            cusparseGetErrorString(error)));                         \
            return;                                                                  \
        }                                                                           \
    } while(0) 

#define WRAPPER_CUSPARSE_CHECK(cmd)                                                 \
    do {                                                                            \
        cusparseStatus_t error = (cmd);                                             \
        if (error != CUSPARSE_STATUS_SUCCESS) {                                     \
            return (tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",       \
                            cusparseGetErrorString(error)));                         \
        }                                                                           \
    } while(0) 

#define PLUGIN_CUBLAS_CHECK(ctx, cmd)                                                                                   \
    do {                                                                                                                \
        cublasStatus_t error = (cmd);                                                                                   \
        switch (error) {                                                                                                \
            case CUBLAS_STATUS_SUCCESS: {break;}                                                                        \
            case CUBLAS_STATUS_NOT_INITIALIZED: {                                                                       \
                ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_NOT_INITIALIZED."));\
                return;                                                                                                 \
            }                                                                                                           \
            case CUBLAS_STATUS_ALLOC_FAILED: {                                                                          \
                ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_ALLOC_FAILED."));   \
                return;                                                                                                 \
            }                                                                                                           \
            case CUBLAS_STATUS_INVALID_VALUE: {                                                                         \
                ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_INVALID_VALUE."));  \
                return;                                                                                                 \
            }                                                                                                           \
            case CUBLAS_STATUS_ARCH_MISMATCH: {                                                                         \
                ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_ARCH_MISMATCH."));  \
                return;                                                                                                 \
            }                                                                                                           \
            case CUBLAS_STATUS_MAPPING_ERROR: {                                                                         \
                ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_MAPPING_ERROR."));  \
                return;                                                                                                 \
            }                                                                                                           \
            case CUBLAS_STATUS_EXECUTION_FAILED: {                                                                      \
                ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_EXECUTION_FAILED."));\
                return;                                                                                                 \
            }                                                                                                           \
            case CUBLAS_STATUS_INTERNAL_ERROR: {                                                                        \
                ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_INTERNAL_ERROR.")); \
                return;                                                                                                 \
            }                                                                                                           \
            case CUBLAS_STATUS_NOT_SUPPORTED: {                                                                         \
                ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_NOT_SUPPORTED."));  \
                return;                                                                                                 \
            }                                                                                                           \
            case CUBLAS_STATUS_LICENSE_ERROR: {                                                                         \
                ctx->SetStatus(tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_LICENSE_ERROR."));  \
                return;                                                                                                 \
            }                                                                                                           \
        }                                                                                                               \
    } while(0)

#define WRAPPER_CUBLAS_CHECK(cmd)                                                                                       \
    do {                                                                                                                \
        cublasStatus_t error = (cmd);                                                                                   \
        switch (error) {                                                                                                \
            case CUBLAS_STATUS_SUCCESS: {break;}                                                                        \
            case CUBLAS_STATUS_NOT_INITIALIZED: {                                                                       \
                return (tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_NOT_INITIALIZED."));       \
            }                                                                                                           \
            case CUBLAS_STATUS_ALLOC_FAILED: {                                                                          \
                return (tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_ALLOC_FAILED."));          \
            }                                                                                                           \
            case CUBLAS_STATUS_INVALID_VALUE: {                                                                         \
                return (tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_INVALID_VALUE."));         \
            }                                                                                                           \
            case CUBLAS_STATUS_ARCH_MISMATCH: {                                                                         \
                return (tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_ARCH_MISMATCH."));         \
            }                                                                                                           \
            case CUBLAS_STATUS_MAPPING_ERROR: {                                                                         \
                return (tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_MAPPING_ERROR."));         \
            }                                                                                                           \
            case CUBLAS_STATUS_EXECUTION_FAILED: {                                                                      \
                return (tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_EXECUTION_FAILED."));      \
            }                                                                                                           \
            case CUBLAS_STATUS_INTERNAL_ERROR: {                                                                        \
                return (tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_INTERNAL_ERROR."));        \
            }                                                                                                           \
            case CUBLAS_STATUS_NOT_SUPPORTED: {                                                                         \
                return (tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_NOT_SUPPORTED."));         \
            }                                                                                                           \
            case CUBLAS_STATUS_LICENSE_ERROR: {                                                                         \
                return (tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " CUBLAS_STATUS_LICENSE_ERROR."));         \
            }                                                                                                           \
        }                                                                                                               \
    } while(0)


struct EraseEmbeddingKeysFunctor {
    virtual void operator()(void* keys_ptr, const int dev_id, const size_t slot_num, const size_t max_nnz,
                            const int gpu_count, const size_t elem_size, const size_t sm_count,
                            cudaStream_t stream) = 0;
};

template <typename T>
struct EraseDistributedEmbeddingKeysFunctor : public EraseEmbeddingKeysFunctor {
    void operator()(void* keys_ptr, const int dev_id, const size_t slot_num, const size_t max_nnz,
                    const int gpu_count, const size_t elem_size, const size_t sm_count,
                    cudaStream_t stream) override;
};

template <typename T>
struct EraseLocalizedEmbeddingKeysFunctor : public EraseEmbeddingKeysFunctor {
    void operator()(void* keys_ptr, const int dev_id, const size_t slot_num, const size_t max_nnz,
                    const int gpu_count, const size_t elem_size, const size_t sm_count,
                    cudaStream_t stream) override;
};


struct ConvertDenseToCSRFunctor {
    virtual void operator()(void* keys_ptr, int row, int col, const cusparseHandle_t& cusparse_handle,
                            const cublasHandle_t& cublas_handle, void* csr_values, 
                            int* csr_row_offsets, int* csr_col_indices,
                            long long* total_nnz, cusparseMatDescr_t& desc, int* nnz_row,
                            void* keys_ptr_transpose) = 0;
};

struct ConvertDenseToCSRDoubleFunctor : public ConvertDenseToCSRFunctor {
    void operator()(void* keys_ptr, int row, int col, const cusparseHandle_t& cusparse_handle,
                    const cublasHandle_t& cublas_handle, void* csr_values, 
                    int* csr_row_offsets, int* csr_col_indices,
                    long long* total_nnz, cusparseMatDescr_t& desc, int* nnz_row,
                    void* keys_ptr_transpose) override;
};

struct ConvertDenseToCSRFloatFunctor : public ConvertDenseToCSRFunctor {
    void operator()(void* keys_ptr, int row, int col, const cusparseHandle_t& cusparse_handle,
                    const cublasHandle_t& cublas_handle, void* csr_values, 
                    int* csr_row_offsets, int* csr_col_indices,
                    long long* total_nnz, cusparseMatDescr_t& desc, int* nnz_row,
                    void* keys_ptr_transpose) override;
};


template <typename T>
void all_keys_plus_1(T* keys_ptr, const size_t elem_size, const size_t sm_count, cudaStream_t stream);

template <typename input_type, typename output_type>
void cast_elements(const input_type* input_ptr, output_type* output_ptr, const size_t elem_size,
                     const size_t sm_count, cudaStream_t stream);

template <typename T>
void erase_distributed_embedding_keys(T* keys_ptr, const int dev_id, const int gpu_count, 
                                      const size_t elem_size, const size_t sm_count, cudaStream_t stream);

template <typename T>
void fuse_keys_plus_erase_distributed_embedding(T* keys_ptr, const int dev_id, const int gpu_count, 
                                    const size_t elem_size, const size_t sm_count, cudaStream_t stream);

template <typename T>
void erase_localized_embedding_keys(T* keys_ptr, const int dev_id, const size_t slot_num, const size_t max_nnz, 
                                    const int gpu_count, const size_t elem_size, const size_t sm_count, 
                                    cudaStream_t stream);

template <typename T>
void convert_dense_to_csr(T* keys_ptr, int row, int col, 
                          const cusparseHandle_t& handle,
                          const cublasHandle_t& cublas_handle,
                          T* csr_values, int* csr_row_offsets, int * csr_col_indices,
                          long long* total_nnz, 
                          cusparseMatDescr_t& desc, 
                          int* nnz_row,
                          T* keys_ptr_transpose);

template <typename T>
void value_tensors_subtract_1(T* values_ptr, const size_t elem_size, const size_t sm_count, cudaStream_t stream);


/** This function is used for localized embedding.
* To select dev-corresponding slots into output_ptr.
* @param input_ptr, original all_keys. and the output_ptr
* @param temp_ptr, temporary storage
* @param elem_size, how many elements
* @param binary_vec_ptr, binary vector, each value represet whether should be selected.
*/
template <typename flag_type>
cudaError_t select_slots(int* input_ptr, int* output_ptr, const size_t elem_size, int* d_temp_storage,
                        flag_type* binary_flag, size_t temp_storage_bytes, int* d_num_selected_out, 
                        cudaStream_t stream);

/** This function is used get binary flag vector for inputs.
* The flag is used decide whether to copy its corresponding value to output.
*/
template <typename T>
void generate_binary_vec(T* input_ptr, const size_t elem_size, const int dest_dev_id, 
                        const size_t slot_num, const int gpu_count, const size_t sm_count, cudaStream_t stream);


/** This function is used to decide the temporary device storage bytes.
*/
template <typename flag_type>
cudaError_t get_temp_storage_bytes(int* input_ptr, flag_type* binary_flag, int* output_ptr,
                                   const size_t elem_size, size_t& temp_storage_bytes);

/** This function is used to get roof of the number by the specified base.
* result = ((input + base - 1) / base) * base; 
*/
size_t num_roof(const size_t number, const size_t base);


template <typename T>
struct CudaAllocator {
    typedef T value_type;
    CudaAllocator() = default;
    T* allocate(size_t n);
    void deallocate(T* ptr, size_t n);
};

template <typename T>
struct CudaHostAllocator {
    typedef T value_type;
    CudaHostAllocator() = default;
    T* allocate(size_t n);
    void deallocate(T* ptr, size_t n);
};


template <typename T>
void print_cuda_ptr(T* dev_ptr, const size_t elem_size);

} // namespace CudaUtils

#endif //CUDA_UTILS_H
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

#define WRAPPER_REQUIRE_OK(cmd)                                             \
    do {                                                                    \
        tensorflow::Status status = (cmd);                                  \
        if (tensorflow::Status::OK() != status) { return status; }          \
    } while (0) 

#define WRAPPER_CUDA_CHECK(cmd)                                                 \
    do {                                                                        \
        cudaError_t error = (cmd);                                              \
        if (error != cudaSuccess) {                                             \
            return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",    \
                            cudaGetErrorString(error));                         \
        }                                                                       \
    } while(0)

#define WRAPPER_NCCL_CHECK(cmd)                                                 \
    do {                                                                        \
        ncclResult_t r = (cmd);                                                 \
        if (ncclSuccess != r) {                                                 \
            return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",    \
                            ncclGetErrorString(r));                             \
        }                                                                       \
    } while (0)

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

template <typename input_type, typename output_type>
void cast_elements(const input_type* input_ptr, output_type* output_ptr, const size_t elem_size,
                     const size_t sm_count, cudaStream_t stream);

/** This function is used to get roof of the number by the specified base.
* result = ((input + base - 1) / base) * base; 
*/
size_t num_roof(const size_t number, const size_t base);


/*warpper of cub::DeviceSelect*/
template <typename input_type, typename flag_type, typename output_type>
cudaError_t cub_flagged(void* d_temp_storage, size_t& temp_storage_bytes, input_type* d_in,
                        flag_type* d_flags, output_type* d_out, size_t* d_num_selected_out,
                        int num_items, cudaStream_t stream = 0, bool debug_synchronous = false);

template <typename input_type>
void distributed_binary_vector(const input_type* input_values, const size_t elem_size, 
                               const size_t gpu_count, const size_t dev_id,
                               bool* binary_out, cudaStream_t stream);
template <typename input_type>
void localized_binary_vector(const input_type* input_row_indices, const size_t elem_size,
                             const size_t gpu_count, const size_t dev_id, const size_t slot_num,
                             bool* binary_out, cudaStream_t stream);

template <typename T>
void localized_new_row_indices(const T* row_indices, T* dev_row_indices, const size_t slot_num, 
                               const size_t dev_slot_num, const size_t gpu_count, const size_t elem_size,
                               cudaStream_t stream);


template <typename T>
void kernel_print(T value, cudaStream_t stream);

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
void print_cuda_ptr(T * dev_ptr, const size_t elem_size);

} // namespace CudaUtils

#endif //CUDA_UTILS_H
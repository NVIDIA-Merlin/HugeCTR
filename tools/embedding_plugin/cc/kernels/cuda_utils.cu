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

#include "cuda_utils.h"
#include <cub/cub/cub.cuh>
#include <memory>

namespace CudaUtils {

/** this function is used to modify keys dense tensor to erase those keys 
* do not belong to that gpu to 0. (0 represet invalid keys, and original 0-key 
* has been mapped to 1.)
* @param kers_ptr, the pointer to 2-D dense keys tensor, corresponding shape is
* [batchsize * slot_num, max_nnz]
* @param dev_id, which dev_id this keys_ptr belongs to.
* @param elem_size, the element size of keys_ptr.
*/
template <typename T>
__global__ void erase_distributed_embedding_keys(T* keys_ptr, const int dev_id, 
                                                 const int gpu_count, const size_t elem_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int strid = blockDim.x * gridDim.x;
    for (size_t i = gid; i < elem_size; i += strid) {
        if (keys_ptr[i] > 0 && ((keys_ptr[i] - 1) % gpu_count != dev_id)) {
            keys_ptr[i] = 0;
        }
    }
}

/** this function is used to modify keys dense tensor to erase those keys 
* do not belong to that gpu to 0.
* @param kers_ptr, the pointer to 2-D dense keys tensor, corresponding shape is
* [batchsize * slot_num, max_nnz]
* @param dev_id, which dev_id this keys_ptr belongs to.
* @param elem_size, the element size of keys_ptr.
* @param batchsize, the batchsize
* @param gpu_count, how many gpus
*/
template <typename T>
__global__ void erase_localized_embedding_keys(T* keys_ptr, const int dev_id,
                                               const size_t slot_num, const size_t max_nnz,
                                               const int gpu_count, const size_t elem_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int strid = blockDim.x * gridDim.x;
    int row_idx = gid / max_nnz;
    int slot_idx = row_idx % slot_num;
    int key_dev_id = slot_idx % gpu_count;
    for (size_t i = gid; i < elem_size; i += strid) {
        if (key_dev_id != dev_id) {
            keys_ptr[i] = 0;
        }
    }
}

/** this function is used to modify the elements.
* @param, the pointer to 2-D tensor keys tensor, corresponding shape is 
* [batchsize * slot_num, max_nnz]
* @param fn, how to modify each element. input element, output its modified value.
* @param elem_size, how many elements.
*/
template <typename input_type, typename Func>
__global__ void modify_elements(input_type* keys_ptr, Func fn, const size_t elem_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int strid = blockDim.x * gridDim.x;
    for (size_t i = gid; i < elem_size; i += strid) {
        keys_ptr[i] = fn(keys_ptr[i]);
    }
}

template <typename Func, typename input_type, typename output_type>
__global__ void modify_elements(const input_type* input_ptr, output_type* output_ptr, Func fn, const size_t elem_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int strid = blockDim.x * gridDim.x;
    for (size_t i = gid; i < elem_size; i += strid) {
        output_ptr[i] = fn(input_ptr[i]);
    }
}

/** This function is used to generate binary vector for csr_row_offset.
* @param elem_size, how many element in input_ptr, and it is equal to the elem_size of csr_row_offset.
* @param dest_dev_id, decide which index should be reside on this destination device. if True, its value is set to 1, 
* otherwise, the value is 0.
*/
template <typename T>
__global__ void generate_binary_vec(T* input_ptr, const size_t elem_size, const int dest_dev_id, 
                                    const size_t slot_num, const int gpu_count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int strid = blockDim.x * gridDim.x;
    for (size_t i = gid; i < elem_size; i += strid) {
        input_ptr[i] = 1; // initialize the value to 1.
        if (i > 0) { // ignore the first value. it should be 1 anyway.
            int row_idx = i - 1;
            int slot_idx = row_idx % slot_num;
            int dev_id = slot_idx % gpu_count;
            input_ptr[i] = (dev_id == dest_dev_id) ? 1 : 0;
        }
    }
}


/** This function fused value + 1 and erase_distributed_embedding_keys
*/
template <typename T>
__global__ void fuse_keys_plus_erase_distributed_embedding(T* keys_ptr, const int dev_id, const int gpu_count, 
                        const size_t elem_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int strid = blockDim.x * gridDim.x;
    for (size_t i = gid; i < elem_size; i += strid) {
        T temp_key = keys_ptr[i];
        temp_key += 1;
        if (temp_key <= 0 || (temp_key - 1) % gpu_count != dev_id) {
            temp_key = 0;
        } 
        keys_ptr[i] = temp_key;
    }
}

template <typename T>
void all_keys_plus_1(T* keys_ptr, const size_t elem_size, const size_t sm_count, cudaStream_t stream) {
    int block_dim = 128;
    // size_t grid_dim = num_roof((elem_size + block_dim - 1) / block_dim, sm_count);
    int grid_dim = (elem_size + block_dim - 1) / block_dim;
    auto fn = [] __device__(T key) -> T { return key + 1; };
    modify_elements<<<grid_dim, block_dim, 0, stream>>>(keys_ptr, fn, elem_size);
}

template <typename input_type, typename output_type>
void cast_elements(const input_type* input_ptr, output_type* output_ptr, const size_t elem_size,
                     const size_t sm_count, cudaStream_t stream) {
    int block_dim = 128;
    // size_t grid_dim = num_roof((elem_size + block_dim - 1) / block_dim, sm_count);
    int grid_dim = (elem_size + block_dim - 1) / block_dim;
    auto cast_fn = [] __device__ (input_type num) -> output_type { return static_cast<output_type>(num); };
    modify_elements<<<grid_dim, block_dim, 0, stream>>>(input_ptr, output_ptr, cast_fn, elem_size);
}


template <typename T>
void erase_distributed_embedding_keys(T* keys_ptr, const int dev_id, const int gpu_count, 
                                      const size_t elem_size, const size_t sm_count, cudaStream_t stream) {
    int block_dim = 128;
    // size_t grid_dim = num_roof((elem_size + block_dim - 1) / block_dim, sm_count);
    int grid_dim = (elem_size + block_dim - 1) / block_dim;
    erase_distributed_embedding_keys<<<grid_dim, block_dim, 0, stream>>>(keys_ptr, dev_id, gpu_count, elem_size);
}

template <typename T>
void erase_localized_embedding_keys(T* keys_ptr, const int dev_id, const size_t slot_num, const size_t max_nnz, 
                                    const int gpu_count, const size_t elem_size, const size_t sm_count, cudaStream_t stream) {
    int block_dim = 128;
    // size_t grid_dim = num_roof((elem_size + block_dim - 1) / block_dim, sm_count);
    int grid_dim = (elem_size + block_dim - 1) / block_dim;
    erase_localized_embedding_keys<<<grid_dim, block_dim, 0, stream>>>(keys_ptr, dev_id, slot_num, 
                                                                        max_nnz, gpu_count, elem_size);
}

template <typename T>
void fuse_keys_plus_erase_distributed_embedding(T* keys_ptr, const int dev_id, const int gpu_count, 
                                    const size_t elem_size, const size_t sm_count, cudaStream_t stream) {
    int block_dim = 128;
    // size_t grid_dim = num_roof((elem_size + block_dim - 1) / block_dim, sm_count);
    int grid_dim = (elem_size + block_dim - 1) / block_dim;
    fuse_keys_plus_erase_distributed_embedding<<<grid_dim, block_dim, 0, stream>>>(keys_ptr, dev_id, gpu_count, elem_size);                                    
}


template <typename T>
void convert_dense_to_csr(T* keys_ptr, int row, int col, 
                          const cusparseHandle_t& handle,
                          const cublasHandle_t& cublas_handle,
                          T* csr_values, int* csr_row_offsets, int * csr_col_indices,
                          long long* total_nnz, 
                          cusparseMatDescr_t& desc, int* nnz_row,
                          T* keys_ptr_transpose) {

    if (std::is_same<T, long long>::value) {
        // transpose
        const double alpha = 1.0;
        const double beta = 0.0;
        cublasStatus_t cublas_error = cublasDgeam(cublas_handle,
                                                CUBLAS_OP_T, /*transa*/
                                                CUBLAS_OP_N, /*transb*/
                                                row, /*number of rows*/
                                                col, /*number of cols*/
                                                &alpha, /*alpha*/
                                                reinterpret_cast<double*>(keys_ptr), /*A*/
                                                col, /*leading dimension*/
                                                &beta, /*beta*/
                                                reinterpret_cast<double*>(keys_ptr), /*B*/
                                                row, /*leading dimension*/
                                                reinterpret_cast<double*>(keys_ptr_transpose), /*C*/
                                                row /*leading dimension*/);
        if (cublas_error != CUBLAS_STATUS_SUCCESS) {
            std::cout << __FILE__ << ":" << __LINE__ << " " << "cublas error: " << cublas_error << std::endl;
            exit(-1);
        }

        int m = row /*row number*/, n = col /*column number*/;  

        int temp_total_nnz = 0;
        cusparseStatus_t status = cusparseDnnz(handle,
                                            CUSPARSE_DIRECTION_ROW, /*count nnz direction*/
                                            m, /*number of rows*/  
                                            n, /*number of columns*/
                                            desc, /*descriptor of matrix*/ 
                                            reinterpret_cast<double*>(keys_ptr_transpose),
                                            m, /*leading dimension*/
                                            nnz_row, /*output*/
                                            &temp_total_nnz);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cout << __FILE__ << ":" << __LINE__ << " " << cusparseGetErrorString(status) << std::endl;
            exit(-1);
        }
        *total_nnz = static_cast<long long>(temp_total_nnz);

        status = cusparseDdense2csr(handle,
                                    m, /*number of rows of matrix A*/
                                    n, /*number of columns of matrix A*/
                                    desc, /*the descriptor of matrix A*/
                                    reinterpret_cast<double*>(keys_ptr_transpose), /*array of dimensions (lda, n)*/
                                    m, /*leading dimension*/
                                    nnz_row, /*nnz array*/
                                    reinterpret_cast<double*>(csr_values), /*csr values*/
                                    csr_row_offsets, /*csr row_offset*/
                                    csr_col_indices/*csr column indices*/);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cout << __FILE__ << ":" << __LINE__ << " " << cusparseGetErrorString(status) << std::endl;
            exit(-1);
        }
    } else if (std::is_same<T, unsigned int>::value) {
        // transpose
        const float alpha = 1.0;
        const float beta = 0.0;
        cublasStatus_t cublas_error = cublasSgeam(cublas_handle,
                                                CUBLAS_OP_T, /*transa*/
                                                CUBLAS_OP_N, /*transb*/
                                                row, /*number of rows*/
                                                col, /*number of cols*/
                                                &alpha, /*alpha*/
                                                reinterpret_cast<float*>(keys_ptr), /*A*/
                                                col, /*leading dimension*/
                                                &beta, /*beta*/
                                                reinterpret_cast<float*>(keys_ptr), /*B*/
                                                row, /*leading dimension*/
                                                reinterpret_cast<float*>(keys_ptr_transpose), /*C*/
                                                row /*leading dimension*/);
        if (cublas_error != CUBLAS_STATUS_SUCCESS) {
            std::cout << __FILE__ << ":" << __LINE__ << " " << "cublas error: " << cublas_error << std::endl;
            exit(-1);
        }

        int m = row /*row number*/, n = col /*column number*/;  

        int temp_total_nnz = 0;
        cusparseStatus_t status = cusparseSnnz(handle,
                                            CUSPARSE_DIRECTION_ROW, /*count nnz direction*/
                                            m, /*number of rows*/  
                                            n, /*number of columns*/
                                            desc, /*descriptor of matrix*/ 
                                            reinterpret_cast<float*>(keys_ptr_transpose),
                                            m, /*leading dimension*/
                                            nnz_row, /*output*/
                                            &temp_total_nnz);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cout << __FILE__ << ":" << __LINE__ << " " << cusparseGetErrorString(status) << std::endl;
            exit(-1);
        }
        *total_nnz = static_cast<long long>(temp_total_nnz);

        status = cusparseSdense2csr(handle,
                                    m, /*number of rows of matrix A*/
                                    n, /*number of columns of matrix A*/
                                    desc, /*the descriptor of matrix A*/
                                    reinterpret_cast<float*>(keys_ptr_transpose), /*array of dimensions (lda, n)*/
                                    m, /*leading dimension*/
                                    nnz_row, /*nnz array*/
                                    reinterpret_cast<float*>(csr_values), /*csr values*/
                                    csr_row_offsets, /*csr row_offset*/
                                    csr_col_indices/*csr column indices*/);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cout << __FILE__ << ":" << __LINE__ << " " << cusparseGetErrorString(status) << std::endl;
            exit(-1);
        }
    }
}

template <typename T>
void value_tensors_subtract_1(T* values_ptr, const size_t elem_size, const size_t sm_count, cudaStream_t stream) {
    int block_dim = 128;
    size_t grid_dim = num_roof((elem_size + block_dim - 1) / block_dim, sm_count);
    auto fn = [] __device__ (T value) { if (value > 0) {return value - 1;} else {return value;} };
    modify_elements<<<grid_dim, block_dim, 0, stream>>>(values_ptr, fn, elem_size);
}


template <typename T>
void generate_binary_vec(T* input_ptr, const size_t elem_size, const int dest_dev_id, 
                        const size_t slot_num, const int gpu_count, const size_t sm_count, cudaStream_t stream) {
    int block_dim = 128;
   size_t grid_dim = num_roof((elem_size + block_dim - 1) / block_dim, sm_count);
    generate_binary_vec<<<grid_dim, block_dim, 0, stream>>>(input_ptr, elem_size, dest_dev_id, 
                                                            slot_num, gpu_count);
}


template <typename flag_type>
cudaError_t get_temp_storage_bytes(int* input_ptr, flag_type* binary_flag, int* output_ptr,
                                   const size_t elem_size, size_t& temp_storage_bytes) {
    int d_num_selected_out = 0;
    cudaError_t error = cub::DeviceSelect::Flagged(nullptr, /*temp storage*/
                                                   temp_storage_bytes, /*temp_storage_bytes*/
                                                   input_ptr, /*d_in*/
                                                   binary_flag, /*d_flags*/
                                                   output_ptr, /*d_out*/
                                                   &d_num_selected_out, /*d_num_selected_out*/
                                                   static_cast<int>(elem_size)/*num_items*/);
    return error;
}


template <typename flag_type>
cudaError_t select_slots(int* input_ptr, int* output_ptr, const size_t elem_size, int* d_temp_storage,
                        flag_type* binary_flag, size_t temp_storage_bytes, int* d_num_selected_out,
                        cudaStream_t stream) {
    cudaError_t error = cudaMemsetAsync(output_ptr, 0, sizeof(int) * elem_size, stream);
    if (error != cudaSuccess) return error;

    error = cub::DeviceSelect::Flagged(d_temp_storage, /*temp storage*/
                                        temp_storage_bytes, /*temp_storage_bytes*/
                                        input_ptr, /*d_in*/
                                        binary_flag, /*d_flags*/
                                        output_ptr, /*d_out*/
                                        d_num_selected_out, /*d_num_selected_out*/
                                        static_cast<int>(elem_size), /*num_items*/
                                        stream, /*cudaStream_t*/
                                        false/*debug_synchronous*/); 
    if (error != cudaSuccess) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(error) << std::endl; 
        return error;
    }
    return cudaSuccess;
}


size_t num_roof(const size_t number, const size_t base) {
    return ((number + base - 1) / base) * base;
}



template <typename T>
void EraseDistributedEmbeddingKeysFunctor<T>::operator()(void* keys_ptr, const int dev_id, const size_t slot_num, 
                                                        const size_t max_nnz, const int gpu_count, const size_t elem_size,
                                                        const size_t sm_count, cudaStream_t stream) {
    
    int block_dim = 128;
    // size_t grid_dim = num_roof((elem_size + block_dim - 1) / block_dim, sm_count);
    int grid_dim = (elem_size + block_dim - 1) / block_dim;
    erase_distributed_embedding_keys<<<grid_dim, block_dim, 0, stream>>>(reinterpret_cast<T*>(keys_ptr), 
                                        dev_id, gpu_count, elem_size);
}

template <typename T>
void EraseLocalizedEmbeddingKeysFunctor<T>::operator()(void* keys_ptr, const int dev_id, const size_t slot_num, 
                                                       const size_t max_nnz, const int gpu_count, const size_t elem_size,
                                                       const size_t sm_count, cudaStream_t stream) {
    int block_dim = 128;
    // size_t grid_dim = num_roof((elem_size + block_dim - 1) / block_dim, sm_count);
    int grid_dim = (elem_size + block_dim - 1) / block_dim;
    erase_localized_embedding_keys<<<grid_dim, block_dim, 0, stream>>>(reinterpret_cast<T*>(keys_ptr), 
                                                                        dev_id, slot_num, 
                                                                        max_nnz, gpu_count, elem_size);
}


void ConvertDenseToCSRDoubleFunctor::operator()(void* keys_ptr, int row, int col, 
                    const cusparseHandle_t& cusparse_handle,
                    const cublasHandle_t& cublas_handle, void* csr_values, 
                    int* csr_row_offsets, int* csr_col_indices,
                    long long* total_nnz, cusparseMatDescr_t& desc, int* nnz_row,
                    void* keys_ptr_transpose) {
    // transpose
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasStatus_t cublas_error = cublasDgeam(cublas_handle,
                                            CUBLAS_OP_T, /*transa*/
                                            CUBLAS_OP_N, /*transb*/
                                            row, /*number of rows*/
                                            col, /*number of cols*/
                                            &alpha, /*alpha*/
                                            reinterpret_cast<double*>(keys_ptr), /*A*/
                                            col, /*leading dimension*/
                                            &beta, /*beta*/
                                            reinterpret_cast<double*>(keys_ptr), /*B*/
                                            row, /*leading dimension*/
                                            reinterpret_cast<double*>(keys_ptr_transpose), /*C*/
                                            row /*leading dimension*/);
    if (cublas_error != CUBLAS_STATUS_SUCCESS) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << "cublas error: " << cublas_error << std::endl;
        exit(-1);
    }

    int m = row /*row number*/, n = col /*column number*/;  

    int temp_total_nnz = 0;
    cusparseStatus_t status = cusparseDnnz(cusparse_handle,
                                        CUSPARSE_DIRECTION_ROW, /*count nnz direction*/
                                        m, /*number of rows*/  
                                        n, /*number of columns*/
                                        desc, /*descriptor of matrix*/ 
                                        reinterpret_cast<double*>(keys_ptr_transpose),
                                        m, /*leading dimension*/
                                        nnz_row, /*output*/
                                        &temp_total_nnz);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cusparseGetErrorString(status) << std::endl;
        exit(-1);
    }
    *total_nnz = static_cast<long long>(temp_total_nnz);

    status = cusparseDdense2csr(cusparse_handle,
                                m, /*number of rows of matrix A*/
                                n, /*number of columns of matrix A*/
                                desc, /*the descriptor of matrix A*/
                                reinterpret_cast<double*>(keys_ptr_transpose), /*array of dimensions (lda, n)*/
                                m, /*leading dimension*/
                                nnz_row, /*nnz array*/
                                reinterpret_cast<double*>(csr_values), /*csr values*/
                                csr_row_offsets, /*csr row_offset*/
                                csr_col_indices/*csr column indices*/);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cusparseGetErrorString(status) << std::endl;
        exit(-1);
    }
}

void ConvertDenseToCSRFloatFunctor::operator()(void* keys_ptr, int row, int col, 
                    const cusparseHandle_t& cusparse_handle,
                    const cublasHandle_t& cublas_handle, void* csr_values, 
                    int* csr_row_offsets, int* csr_col_indices,
                    long long* total_nnz, cusparseMatDescr_t& desc, int* nnz_row,
                    void* keys_ptr_transpose) {
    // transpose
    const float alpha = 1.0;
    const float beta = 0.0;
    cublasStatus_t cublas_error = cublasSgeam(cublas_handle,
                                            CUBLAS_OP_T, /*transa*/
                                            CUBLAS_OP_N, /*transb*/
                                            row, /*number of rows*/
                                            col, /*number of cols*/
                                            &alpha, /*alpha*/
                                            reinterpret_cast<float*>(keys_ptr), /*A*/
                                            col, /*leading dimension*/
                                            &beta, /*beta*/
                                            reinterpret_cast<float*>(keys_ptr), /*B*/
                                            row, /*leading dimension*/
                                            reinterpret_cast<float*>(keys_ptr_transpose), /*C*/
                                            row /*leading dimension*/);
    if (cublas_error != CUBLAS_STATUS_SUCCESS) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << "cublas error: " << cublas_error << std::endl;
        exit(-1);
    }

    int m = row /*row number*/, n = col /*column number*/;  

    int temp_total_nnz = 0;
    cusparseStatus_t status = cusparseSnnz(cusparse_handle,
                                        CUSPARSE_DIRECTION_ROW, /*count nnz direction*/
                                        m, /*number of rows*/  
                                        n, /*number of columns*/
                                        desc, /*descriptor of matrix*/ 
                                        reinterpret_cast<float*>(keys_ptr_transpose),
                                        m, /*leading dimension*/
                                        nnz_row, /*output*/
                                        &temp_total_nnz);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cusparseGetErrorString(status) << std::endl;
        exit(-1);
    }
    *total_nnz = static_cast<long long>(temp_total_nnz);

    status = cusparseSdense2csr(cusparse_handle,
                                m, /*number of rows of matrix A*/
                                n, /*number of columns of matrix A*/
                                desc, /*the descriptor of matrix A*/
                                reinterpret_cast<float*>(keys_ptr_transpose), /*array of dimensions (lda, n)*/
                                m, /*leading dimension*/
                                nnz_row, /*nnz array*/
                                reinterpret_cast<float*>(csr_values), /*csr values*/
                                csr_row_offsets, /*csr row_offset*/
                                csr_col_indices/*csr column indices*/);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cusparseGetErrorString(status) << std::endl;
        exit(-1);
    }
}



template <typename T>
T* CudaAllocator<T>::allocate(size_t n) {
    T* result = nullptr;
    result = static_cast<T*>(malloc(n * sizeof(T)));
    if (!result) throw std::bad_alloc();
    return result;
}

template <typename T>
T* CudaHostAllocator<T>::allocate(size_t n) {
    T* result = nullptr;
    result = static_cast<T*>(malloc(n * sizeof(T)));
    if (!result) throw std::bad_alloc();
    return result;
}

template <typename T>
void CudaAllocator<T>::deallocate(T* ptr, size_t n) {
    if (ptr) {
        for (size_t i = 0; i < n; ++i) {
            if (ptr[i]) {
                cudaFree(ptr[i]);
                ptr[i] = nullptr;
            }
        }
        free(ptr);
        ptr = nullptr;
    }
}

template <typename T>
void CudaHostAllocator<T>::deallocate(T* ptr, size_t n) {
    if (ptr) {
        for (size_t i = 0; i < n; ++i) {
            if (ptr[i]) {
                cudaFreeHost(ptr[i]);
                ptr[i] = nullptr;
            }
        }
        free(ptr);
        ptr = nullptr;
    }
}


template <typename T>
void print_cuda_ptr(T* dev_ptr, const size_t elem_size) {
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
    std::unique_ptr<T []> host_vector(new T[elem_size]());
    error = cudaMemcpy(host_vector.get(), dev_ptr, sizeof(T) * elem_size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }

    for (size_t i = 0; i < elem_size; ++i) {
        std::cout << host_vector[i] << ", " << std::flush;
    }
    std::cout << std::endl;

    return;
}


template class CudaAllocator<int*>;
template class CudaAllocator<long long*>;
template class CudaAllocator<char*>;
template class CudaAllocator<unsigned int*>;

template struct EraseDistributedEmbeddingKeysFunctor<long long>;
template struct EraseLocalizedEmbeddingKeysFunctor<long long>;


template void all_keys_plus_1(long long*, const size_t, const size_t, cudaStream_t);
template void all_keys_plus_1(unsigned int*, const size_t, const size_t, cudaStream_t);
template void erase_distributed_embedding_keys(long long*, const int, const int, const size_t, const size_t, cudaStream_t);
template void erase_distributed_embedding_keys(unsigned int*, const int, const int, const size_t, const size_t, cudaStream_t);
template void erase_localized_embedding_keys(long long*, const int, const size_t, const size_t, const int, const size_t, 
                                            const size_t, cudaStream_t);
template void erase_localized_embedding_keys(unsigned int*, const int, const size_t, const size_t, const int, const size_t, 
                                            const size_t, cudaStream_t);
template void convert_dense_to_csr(long long*, int, int, const cusparseHandle_t&, const cublasHandle_t&, long long*, int*, int*,
                                    long long*, cusparseMatDescr_t&, int*, long long*);
template void convert_dense_to_csr(unsigned int*, int, int, const cusparseHandle_t&, const cublasHandle_t&, unsigned int*, int*,
                                    int*, long long*, cusparseMatDescr_t&, int*, unsigned int*);
template void value_tensors_subtract_1(long long*, const size_t, const size_t, cudaStream_t);
template void value_tensors_subtract_1(unsigned int*, const size_t, const size_t, cudaStream_t);
template void cast_elements(const int*, long long*, const size_t, const size_t, cudaStream_t);
template void generate_binary_vec(char*, const size_t, const int, const size_t, const int, const size_t, cudaStream_t);
template void generate_binary_vec(int*, const size_t, const int, const size_t, const int, const size_t, cudaStream_t);
template cudaError_t select_slots(int*, int*, const size_t, int*, int*, size_t, int*, cudaStream_t);
template cudaError_t get_temp_storage_bytes(int*, int*, int*, const size_t, size_t&);
template void fuse_keys_plus_erase_distributed_embedding(long long*, const int, const int, const size_t, const size_t, cudaStream_t);
template void fuse_keys_plus_erase_distributed_embedding(unsigned int*, const int, const int, const size_t, const size_t, cudaStream_t);

template void print_cuda_ptr(long long*, const size_t);
template void print_cuda_ptr(unsigned int*, const size_t);
template void print_cuda_ptr(int*, const size_t);


} // namespace CudaUtils


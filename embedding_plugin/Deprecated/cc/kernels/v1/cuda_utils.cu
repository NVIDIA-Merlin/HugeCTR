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
#include <cub/cub.cuh>
#include <memory>

namespace CudaUtils {


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


template <typename Func, typename input_type>
__global__ void binary_vector(const input_type* input, const size_t elem_size, 
                              const size_t gpu_count, const size_t dev_id,
                              Func fn, bool* binary_out, const size_t slot_num = 0) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int strid = blockDim.x * gridDim.x;
    for (size_t i = gid; i < elem_size; i += strid) {
        binary_out[i] = (fn(input[i], gpu_count, dev_id, slot_num) ? true : false);
    }
}

template <typename T>
__global__ void localized_new_row_indices(const T* row_indices, T* dev_row_indices, 
                                          const size_t slot_num, const size_t dev_slot_num,
                                          const size_t gpu_count, const size_t elem_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int strid = blockDim.x * gridDim.x;
    T batch_idx = 0, slot_idx = 0, dev_slot_idx = 0;
    for (size_t i = gid; i < elem_size; i += strid){
        batch_idx = row_indices[i] / slot_num;
        slot_idx = row_indices[i] % slot_num;
        dev_slot_idx = slot_idx / gpu_count;
        dev_row_indices[i] = static_cast<T>(batch_idx * dev_slot_num + dev_slot_idx);
    }
}


template <typename T>
__global__ void kernel_print(T value) {
    printf("%d\n", value);
}

template <typename T>
void kernel_print(T value, cudaStream_t stream) {
    kernel_print<<<1, 1, 0, stream>>>(value);
}


template <typename input_type>
void distributed_binary_vector(const input_type* input_values, const size_t elem_size, 
                               const size_t gpu_count, const size_t dev_id,
                               bool* binary_out, cudaStream_t stream) {
    int block_dim = 128;
    int grid_dim = (elem_size + block_dim - 1) / block_dim;
    auto fn = [] __device__(const input_type value, const size_t gpu_count, 
                            const size_t dev_id, const size_t slot_num) -> bool {
        return ((dev_id == value % gpu_count) ? true : false);
    };
    binary_vector<<<grid_dim, block_dim, 0, stream>>>(input_values, elem_size, gpu_count, dev_id, fn, binary_out);
}


template <typename input_type>
void localized_binary_vector(const input_type* input_row_indices, const size_t elem_size,
                             const size_t gpu_count, const size_t dev_id, const size_t slot_num,
                             bool* binary_out, cudaStream_t stream){
    int block_dim = 128;
    int grid_dim = (elem_size + block_dim - 1) / block_dim;
    auto fn = [] __device__(const input_type row_indice, const size_t gpu_count, 
                            const size_t dev_id, const size_t slot_num) -> bool {
        input_type slot_idx = row_indice % slot_num;
        return ((dev_id == slot_idx % gpu_count) ? true : false);
    };
    binary_vector<<<grid_dim, block_dim, 0, stream>>>(input_row_indices, elem_size, gpu_count, dev_id, fn, binary_out, slot_num);
}


template <typename T>
void localized_new_row_indices(const T* row_indices, T* dev_row_indices, const size_t slot_num, 
                               const size_t dev_slot_num, const size_t gpu_count, const size_t elem_size,
                               cudaStream_t stream) {
    int block_dim = 128;
    int grid_dim = (elem_size + block_dim - 1) / block_dim;
    localized_new_row_indices<<<grid_dim, block_dim, 0, stream>>>(row_indices, dev_row_indices, slot_num, 
                                                                 dev_slot_num, gpu_count, elem_size);
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


/*get the roof of the number*/
size_t num_roof(const size_t number, const size_t base) {
    return ((number + base - 1) / base) * base;
}


/*warpper of cub::DeviceSelect*/
template <typename input_type, typename flag_type, typename output_type>
cudaError_t cub_flagged(void* d_temp_storage, size_t& temp_storage_bytes, input_type* d_in,
                        flag_type* d_flags, output_type* d_out, size_t* d_num_selected_out,
                        int num_items, cudaStream_t stream, bool debug_synchronous) {
    return cub::DeviceSelect::Flagged(d_temp_storage, 
                                      temp_storage_bytes,
                                      d_in,
                                      d_flags,
                                      d_out,
                                      d_num_selected_out,
                                      num_items,
                                      stream,
                                      debug_synchronous);
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
template class CudaAllocator<bool*>;
template class CudaAllocator<size_t*>;
template class CudaAllocator<void*>;
template class CudaHostAllocator<size_t*>;

template cudaError_t cub_flagged(void* d_temp_storage, size_t& temp_storage_bytes, long long* d_in,
                        bool* d_flags, int* d_out, size_t* d_num_selected_out,
                        int num_items, cudaStream_t stream, bool debug_synchronous);
template cudaError_t cub_flagged(void* d_temp_storage, size_t& temp_storage_bytes, long long* d_in,
                        bool* d_flags, long long* d_out, size_t* d_num_selected_out,
                        int num_items, cudaStream_t stream, bool debug_synchronous);
template cudaError_t cub_flagged(void* d_temp_storage, size_t& temp_storage_bytes, unsigned int* d_in,
                        bool* d_flags, unsigned int* d_out, size_t* d_num_selected_out,
                        int num_items, cudaStream_t stream, bool debug_synchronous);

template void localized_new_row_indices(const int* row_indices, int* dev_row_indices, const size_t slot_num, 
                                        const size_t dev_slot_num, const size_t gpu_count, const size_t elem_size,
                                        cudaStream_t stream);

template void distributed_binary_vector(const long long* input_values, const size_t elem_size, 
                               const size_t gpu_count, const size_t dev_id,
                               bool* binary_out, cudaStream_t stream);
template void distributed_binary_vector(const unsigned int* input_values, const size_t elem_size, 
                               const size_t gpu_count, const size_t dev_id,
                               bool* binary_out, cudaStream_t stream);

template void localized_binary_vector(const long long* input_row_indices, const size_t elem_size,
                             const size_t gpu_count, const size_t dev_id, const size_t slot_num,
                             bool* binary_out, cudaStream_t stream);

template void cast_elements(const int*, long long*, const size_t, const size_t, cudaStream_t);
template void cast_elements(const int*, unsigned int*, const size_t, const size_t, cudaStream_t);

template void kernel_print(size_t value, cudaStream_t stream);

template void print_cuda_ptr(long long *, const size_t);
template void print_cuda_ptr(unsigned int *, const size_t);
template void print_cuda_ptr(int *, const size_t);
template void print_cuda_ptr(bool *, const size_t);


} // namespace CudaUtils


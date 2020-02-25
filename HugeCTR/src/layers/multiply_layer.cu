/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/layers/multiply_layer.hpp"

#include "HugeCTR/include/layers/element_wise_function.hpp"

#include <algorithm>
#include <functional>
#include "HugeCTR/include/utils.hpp"
#ifndef NDEBUG
#include <iostream>
#endif
 
namespace HugeCTR {
 
namespace {

#define BLOCK_DIM_SIZE 32

template<typename T>
__global__ void multiply_kernel(const T * input, 
                                const T * weight, 
                                T * output, 
                                const int batch_size, 
                                const int vector_length) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t size = (size_t)batch_size * vector_length;

  if(tid < size) {
    output[tid] = input[tid] * weight[tid % vector_length];
  }
}

template<typename T>
__global__ void multiply_transpose_fuse_kernel(const int row,
                                              const int col,
                                              const T * input_1,
                                              const T * input_2,
                                              T * output) {
  __shared__ T sh_data[BLOCK_DIM_SIZE+1][BLOCK_DIM_SIZE];

  unsigned int src_index_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int src_index_y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((src_index_x < col) && (src_index_y < row))
  {
    unsigned int index_in = src_index_y * col + src_index_x;
    sh_data[threadIdx.x][threadIdx.y] = input_1[index_in] * input_2[index_in];
  }

  __syncthreads();

  unsigned int dst_index_x = blockIdx.y*blockDim.y + threadIdx.x;
  unsigned int dst_index_y = blockIdx.x*blockDim.x + threadIdx.y;
  if ((dst_index_x < row) && (dst_index_y < col))
  {
    unsigned int index_out = dst_index_y * row + dst_index_x;
    output[index_out] = sh_data[threadIdx.y][threadIdx.x];
  }
}

// sum reduce computation in one block
template<typename T>
__global__ void sum_reduce_batch_kernel(const int row, // row=blockDim.x
                      const int col,
                      const T * input, 
                      T * output) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  extern __shared__ T sh_data[];
  T sum = 0.0f;
  T sum_total = 0.0f;

  sh_data[tid] = 0.0f;
  __syncthreads();

  for(int gid = tid; gid < col; gid += blockDim.x) {
    sh_data[tid] = input[bid * col + gid];
    __syncthreads();

    for(int num = (blockDim.x/2); num >= warpSize; num /=2) {
      if(tid < num) {
        sum = sh_data[tid] + sh_data[tid + num];
      }
      __syncthreads();

      if(tid < num) {
        sh_data[tid] = sum;
      }
      __syncthreads();
    }

    // sum reduce in a warp
    if(tid < warpSize) {
      sum = sh_data[tid];
      __syncwarp();

      for(int i = (warpSize/2); i >= 1; i /= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, i, warpSize);
      }

      sum_total += sum;
    }

    __syncthreads();
  }

  if(tid == 0) {
    output[bid] += sum_total; // should be "+=" since we have regularization
  }
}

template<typename T>
__global__ void multiply_dgrad_kernel(const T * top_grad,
                                      const T * weight,
                                      T * dgrad,
                                      const int batch_size,
                                      const int vector_length) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t size = (size_t)batch_size * vector_length;

  if(tid < size) {
    dgrad[tid] = top_grad[tid] * weight[tid % vector_length];
  }
}

__host__ inline int GetBlockSize(int size) 
{
  if(size <= 32) {
    return 32;
  }
  else if(size <= 64) {
    return 64;
  }
  else if(size <= 128) {
    return 128;
  }
  else if(size <= 256) {
    return 256;
  }
  else if(size <= 512) {
    return 512;
  }
  else {
    return 1024;
  }
}

template<typename T>
void multiply_wgrad(const T * top_grad,
                    const T * input,
                    T * wgrad,
                    T * wgrad_tmp_trans,
                    const int batch_size,
                    const int vector_length,
                    cudaStream_t stream) {

  dim3 blockSize1(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE, 1);
  dim3 gridSize1((vector_length+blockSize1.x-1)/blockSize1.x, (batch_size+blockSize1.y-1)/blockSize1.y, 1);
  multiply_transpose_fuse_kernel<<<gridSize1, blockSize1, 0, stream>>>(batch_size, 
                                                                      vector_length,
                                                                      top_grad,
                                                                      input,
                                                                      wgrad_tmp_trans);

  dim3 blockSize2(GetBlockSize(batch_size) , 1, 1);
  dim3 gridSize2(vector_length, 1, 1);
  sum_reduce_batch_kernel<<<gridSize2, blockSize2, blockSize2.x*sizeof(float), stream>>>(vector_length, 
                                                                                        batch_size, 
                                                                                        wgrad_tmp_trans, 
                                                                                        wgrad);
}

template<typename T> 
void multiply_dgrad(const T * top_grad,
                    const T * weight,
                    T * dgrad,
                    const int batch_size,
                    const int vector_length,
                    cudaStream_t stream) {

  size_t size = (size_t)batch_size * vector_length;

  dim3 blockSize(64, 1, 1);
  dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1, 1);
  multiply_dgrad_kernel<<<gridSize, blockSize, 0, stream>>>(top_grad, weight, dgrad, 
                                                            batch_size, vector_length);
}

}

MultiplyLayer::MultiplyLayer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                            const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
                            const std::shared_ptr<Tensor<float>>& in_tensor,
                            const std::shared_ptr<Tensor<float>>& out_tensor, 
                            int device_id)
     : Layer(device_id) {
  try {
    CudaDeviceContext context(get_device_id());

    auto in_dims = in_tensor->get_dims();
    if(in_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "Only 2D tensors can be multiplied");
    }
    if(in_tensor->get_format() != TensorFormat_t::HW) {
      CK_THROW_(Error_t::WrongInput, "Only TensorFormat_t::HW is allowed for multiply layer");
    }
    auto out_dims = out_tensor->get_dims();
    if(out_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "only 2D tensors can be set as the result of multiply layer");
    }
    if(out_tensor->get_format() != TensorFormat_t::HW) {
      CK_THROW_(Error_t::WrongInput, "Only TensorFormat_t::HW is allowed for multiply layer");
    }
    if(get_size_from_dims(in_dims) != get_size_from_dims(out_dims)) {
      std::cout << "in_size=" << get_size_from_dims(in_dims) << ", out_size=" << get_size_from_dims(out_dims) << std::endl;
      CK_THROW_(Error_t::WrongInput, "the size of the output is not the same as the input");   
    }

    in_tensors_.emplace_back(in_tensor);
    out_tensors_.emplace_back(out_tensor);

    std::vector<int> w_dim = {1, in_dims[1]};
    TensorFormat_t w_format = TensorFormat_t::HW;
    weights_.emplace_back(new Tensor<float>(w_dim, weight_buff, w_format));
    wgrad_.emplace_back(new Tensor<float>(w_dim, wgrad_buff, w_format));

    internal_buff_.reset(new GeneralBuffer<float>());
    wgrad_tmp_trans_.reset(new Tensor<float>(in_dims, internal_buff_, w_format));
    internal_buff_->init(get_device_id());

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}
 
void MultiplyLayer::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  float* input = in_tensors_[0]->get_ptr();
  float * weight = weights_[0]->get_ptr();
  float* output = out_tensors_[0]->get_ptr();
  int batch_size = in_tensors_[0]->get_dims()[0];
  int vector_length = in_tensors_[0]->get_dims()[1];
  size_t size = (size_t)batch_size * vector_length;

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size + blockSize.x - 1)/blockSize.x, 1, 1);
  multiply_kernel<<<gridSize, blockSize, 0, stream>>>(input, weight, output, 
                                                      batch_size, vector_length);
}
 
void MultiplyLayer::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  float* weight = weights_[0]->get_ptr();
  float* wgrad = wgrad_[0]->get_ptr();
  float* wgrad_tmp_trans = wgrad_tmp_trans_->get_ptr();
  float* input = in_tensors_[0]->get_ptr();
  float* output = out_tensors_[0]->get_ptr();
  int batch_size = in_tensors_[0]->get_dims()[0];
  int vector_length = in_tensors_[0]->get_dims()[1];

  cudaMemsetAsync(wgrad, 0, wgrad_[0]->get_size(), stream);

  multiply_wgrad(output, input, wgrad, wgrad_tmp_trans, batch_size, vector_length, stream);

  // CAUSION: dgrad computation will modify the "input", so it must be put after wgrad computation
  multiply_dgrad(output, weight, input, batch_size, vector_length, stream);
}
 
}  // namespace HugeCTR
 
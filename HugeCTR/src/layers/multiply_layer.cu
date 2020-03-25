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

#include "HugeCTR/include/layers/multiply_layer.hpp"
#include "HugeCTR/include/layers/element_wise_function.hpp"
#include "HugeCTR/include/utils.hpp"
#include "HugeCTR/include/utils.cuh"

#include <algorithm>
#include <functional>

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
                                int batch_size, 
                                int slot_num,
                                int embedding_vec_size) {
  if((blockIdx.x < batch_size) && (threadIdx.x < embedding_vec_size)) {
    for(int i = 0; i < slot_num; i++) {
      output[blockIdx.x * slot_num * embedding_vec_size + i * embedding_vec_size + threadIdx.x] = 
        input[blockIdx.x * slot_num + i] * weight[i * embedding_vec_size + threadIdx.x];
    }
  }
}

template<typename T>
__global__ void multiply_transpose_fuse_kernel(int batch_size,
                                              int slot_num,
                                              int embedding_vec_size,
                                              const T * top_grad,
                                              const T * input,
                                              T * wgrad_tmp_trans) {
  int row = batch_size;
  int col = slot_num * embedding_vec_size;
  __shared__ T sh_data[BLOCK_DIM_SIZE+1][BLOCK_DIM_SIZE];

  int src_index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int src_index_y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((src_index_x < col) && (src_index_y < row))
  {
    int index_in = src_index_y * col + src_index_x;
    sh_data[threadIdx.x][threadIdx.y] = 
      top_grad[index_in] * input[index_in/embedding_vec_size];
  }

  __syncthreads();

  int dst_index_x = blockIdx.y*blockDim.y + threadIdx.x;
  int dst_index_y = blockIdx.x*blockDim.x + threadIdx.y;
  if ((dst_index_x < row) && (dst_index_y < col))
  {
    int index_out = dst_index_y * row + dst_index_x;
    wgrad_tmp_trans[index_out] = sh_data[threadIdx.y][threadIdx.x];
  }
}

// sum reduce computation in one block
template<typename T>
__global__ void sum_reduce_batch_kernel(int row, // row=gridDim.x
                                        int col,
                                        const T * input, 
                                        T * output) {
  float local_sum = 0.0f;
  for (int tid = threadIdx.x; tid < col; tid += blockDim.x) {
    local_sum += input[blockIdx.x * col + tid];
  }
  __syncthreads();

  local_sum = blockReduceSum(local_sum);
  if (threadIdx.x == 0) {
    output[blockIdx.x] += local_sum;
  }
}

template<typename T>
__global__ void multiply_dgrad_kernel(const T * top_grad,
                                      const T * weight,
                                      T * dgrad,
                                      int batch_size, 
                                      int slot_num,
                                      int embedding_vec_size) {
  if((blockIdx.x < batch_size) && (threadIdx.x < embedding_vec_size)) {
    for(int i = 0; i < slot_num; i++) {
      T local_sum = 
      top_grad[blockIdx.x * slot_num * embedding_vec_size + i * embedding_vec_size + threadIdx.x]
        * weight[i * embedding_vec_size + threadIdx.x];

      local_sum = blockReduceSum(local_sum);
      if(threadIdx.x == 0) {
        dgrad[blockIdx.x * slot_num + i] = local_sum;
      }
    }
  }
}

template<typename T>
void multiply_wgrad(const T * top_grad,
                    const T * input,
                    T * wgrad,
                    T * wgrad_tmp_trans,
                    int batch_size,
                    int slot_num,
                    int embedding_vec_size,
                    cudaStream_t stream) {

  dim3 blockSize1(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE, 1);
  dim3 gridSize1((slot_num*embedding_vec_size+blockSize1.x-1)/blockSize1.x,
                 (batch_size+blockSize1.y-1)/blockSize1.y, 1);
  multiply_transpose_fuse_kernel<<<gridSize1, blockSize1, 0, stream>>>(batch_size, 
                                                                      slot_num,
                                                                      embedding_vec_size,
                                                                      top_grad,
                                                                      input,
                                                                      wgrad_tmp_trans);
             
  dim3 blockSize2(256, 1, 1);
  dim3 gridSize2(slot_num*embedding_vec_size, 1, 1);
  sum_reduce_batch_kernel<<<gridSize2, blockSize2, 0, stream>>>(slot_num*embedding_vec_size, 
                                                                batch_size, 
                                                                wgrad_tmp_trans, 
                                                                wgrad);
}

template<typename T> 
void multiply_dgrad(const T * top_grad,
                    const T * weight,
                    T * dgrad,
                    int batch_size,
                    int slot_num,
                    int embedding_vec_size,
                    cudaStream_t stream) {

  dim3 blockSize(embedding_vec_size, 1, 1);
  dim3 gridSize(batch_size, 1, 1);
  multiply_dgrad_kernel<<<gridSize, blockSize, 0, stream>>>(top_grad, weight, dgrad, 
                                                            batch_size, slot_num,
                                                            embedding_vec_size);
}

} // end of namespace

MultiplyLayer::MultiplyLayer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                            const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
                            const std::shared_ptr<Tensor<float>>& in_tensor,
                            const std::shared_ptr<Tensor<float>>& out_tensor, 
                            int device_id)
     : Layer(device_id) {
  try {
    CudaDeviceContext context(get_device_id());

    auto in_dims = in_tensor->get_dims();
    if(in_dims.size() != 3) {
      CK_THROW_(Error_t::WrongInput, "Only 3D tensors can be multiplied");
    }
    if(in_tensor->get_format() != TensorFormat_t::HSW) {
      CK_THROW_(Error_t::WrongInput, "Only TensorFormat_t::HSW is allowed for multiply layer");
    }
    auto out_dims = out_tensor->get_dims();
    if(out_dims.size() != 3) {
      CK_THROW_(Error_t::WrongInput, "only 3D tensors can be set as the result of multiply layer");
    }
    if(out_tensor->get_format() != TensorFormat_t::HSW) {
      CK_THROW_(Error_t::WrongInput, "Only TensorFormat_t::HSW is allowed for multiply layer");
    }
    if(out_dims[2] > 1024) { // embedding_vec_size 
      CK_THROW_(Error_t::WrongInput, "the out_dims[2] can not be more than 1024 in multiply layer");
    }

    in_tensors_.emplace_back(in_tensor);
    out_tensors_.emplace_back(out_tensor);

    std::vector<int> w_dim = {out_dims[1], out_dims[2]};  // {slot_num. embedding_vec_size}
    TensorFormat_t w_format = TensorFormat_t::HW;
    weights_.emplace_back(new Tensor<float>(w_dim, weight_buff, w_format));
    wgrad_.emplace_back(new Tensor<float>(w_dim, wgrad_buff, w_format));

    internal_buff_.reset(new GeneralBuffer<float>());
    wgrad_tmp_trans_.reset(new Tensor<float>(out_dims, internal_buff_, TensorFormat_t::HSW));
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
  int batch_size = out_tensors_[0]->get_dims()[0];
  int slot_num = out_tensors_[0]->get_dims()[1];
  int embedding_vec_size = out_tensors_[0]->get_dims()[2];

  dim3 blockSize(embedding_vec_size, 1, 1);
  dim3 gridSize(batch_size, 1, 1);
  multiply_kernel<<<gridSize, blockSize, 0, stream>>>(input, weight, output, 
                                                      batch_size, slot_num, 
                                                      embedding_vec_size);
}
 
void MultiplyLayer::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  float* weight = weights_[0]->get_ptr();
  float* wgrad = wgrad_[0]->get_ptr();
  float* wgrad_tmp_trans = wgrad_tmp_trans_->get_ptr();
  float* input = in_tensors_[0]->get_ptr();
  float* output = out_tensors_[0]->get_ptr();
  int batch_size = out_tensors_[0]->get_dims()[0];
  int slot_num = out_tensors_[0]->get_dims()[1];
  int embedding_vec_size = out_tensors_[0]->get_dims()[2];

  cudaMemsetAsync(wgrad, 0, wgrad_[0]->get_size(), stream);

  multiply_wgrad(output, input, wgrad, wgrad_tmp_trans, batch_size, slot_num, embedding_vec_size, stream);

  // CAUSION: dgrad computation will modify the "input", so it must be put after wgrad computation
  multiply_dgrad(output, weight, input, batch_size, slot_num, embedding_vec_size, stream);
}
 
}  // namespace HugeCTR
 
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

#include <algorithm>
#include <functional>
#include <layers/element_wise_function.hpp>
#include <layers/multiply_layer.hpp>
#include <linalg/matrix_vector_op.cuh>
#include <linalg/reduce.cuh>
#include <utils.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

#define BLOCK_DIM_SIZE 32

template <typename T>
__global__ void multiply_kernel(const T* input, const T* weight, T* output, int batch_size,
                                int slot_num, int embedding_vec_size) {
  if ((blockIdx.x < batch_size) && (threadIdx.x < embedding_vec_size)) {
    for (int i = 0; i < slot_num; i++) {
      output[blockIdx.x * slot_num * embedding_vec_size + i * embedding_vec_size + threadIdx.x] =
          input[blockIdx.x * slot_num + i] * weight[i * embedding_vec_size + threadIdx.x];
    }
  }
}

template <typename T>
__global__ void multiply_transpose_fuse_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                               const T* top_grad, const T* input,
                                               T* wgrad_tmp_trans) {
  int row = batch_size;
  int col = slot_num * embedding_vec_size;
  __shared__ T sh_data[BLOCK_DIM_SIZE + 1][BLOCK_DIM_SIZE];

  int src_index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int src_index_y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((src_index_x < col) && (src_index_y < row)) {
    int index_in = src_index_y * col + src_index_x;
    sh_data[threadIdx.x][threadIdx.y] = top_grad[index_in] * input[index_in / embedding_vec_size];
  }

  __syncthreads();

  int dst_index_x = blockIdx.y * blockDim.y + threadIdx.x;
  int dst_index_y = blockIdx.x * blockDim.x + threadIdx.y;
  if ((dst_index_x < row) && (dst_index_y < col)) {
    int index_out = dst_index_y * row + dst_index_x;
    wgrad_tmp_trans[index_out] = sh_data[threadIdx.y][threadIdx.x];
  }
}

template <typename T>
__global__ void multiply_dgrad_kernel(const T* top_grad, const T* weight, T* dgrad, int batch_size,
                                      int slot_num, int embedding_vec_size) {
  if ((blockIdx.x < batch_size) && (threadIdx.x < embedding_vec_size)) {
    for (int i = 0; i < slot_num; i++) {
      T local_sum = top_grad[blockIdx.x * slot_num * embedding_vec_size + i * embedding_vec_size +
                             threadIdx.x] *
                    weight[i * embedding_vec_size + threadIdx.x];

      local_sum = blockReduceSum(local_sum);
      if (threadIdx.x == 0) {
        dgrad[blockIdx.x * slot_num + i] = local_sum;
      }
    }
  }
}

template <typename T>
void multiply_wgrad(const T* top_grad, const T* input, T* wgrad, T* wgrad_tmp_trans, int batch_size,
                    int slot_num, int embedding_vec_size, cudaStream_t stream) {
  dim3 blockSize1(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE, 1);
  dim3 gridSize1((slot_num * embedding_vec_size + blockSize1.x - 1) / blockSize1.x,
                 (batch_size + blockSize1.y - 1) / blockSize1.y, 1);
  multiply_transpose_fuse_kernel<<<gridSize1, blockSize1, 0, stream>>>(
      batch_size, slot_num, embedding_vec_size, top_grad, input, wgrad_tmp_trans);

  MLCommon::LinAlg::reduce(wgrad, wgrad_tmp_trans, batch_size, slot_num * embedding_vec_size,
                           float(0), true, true, stream, true);
}

template <typename T>
void multiply_dgrad(const T* top_grad, const T* weight, T* dgrad, int batch_size, int slot_num,
                    int embedding_vec_size, cudaStream_t stream) {
  dim3 blockSize(embedding_vec_size, 1, 1);  // note that embedding_vec_size should be < 1024
  dim3 gridSize(batch_size, 1, 1);
  multiply_dgrad_kernel<<<gridSize, blockSize, 0, stream>>>(top_grad, weight, dgrad, batch_size,
                                                            slot_num, embedding_vec_size);
}

}  // end of namespace

MultiplyLayer::MultiplyLayer(const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                             const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                             const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blob_buff,
                             const Tensor2<float>& in_tensor, Tensor2<float>& out_tensor,
                             const std::vector<size_t>& weight_dims,
                             const std::shared_ptr<GPUResource>& gpu_resource,
                             std::vector<Initializer_t> initializer_types)
    : Layer(gpu_resource, initializer_types) {
  try {
    const auto& in_dims = in_tensor.get_dimensions();
    if (in_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "Only 2D tensors can be multiplied");
    }
    if (weight_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "Only 2D weights is allowed for multiply layer");
    }
    if (weight_dims[0] != in_dims[1]) {
      CK_THROW_(Error_t::WrongInput, "weight_dims[0] must be equal to in_dims[1]");
    }

    batch_size_ = in_dims[0];
    slot_num_ = weight_dims[0];
    embedding_vec_size_ = weight_dims[1];

    std::vector<size_t> out_dims{batch_size_, slot_num_ * embedding_vec_size_};
    blob_buff->reserve(out_dims, &out_tensor);
    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);

    {
      Tensor2<float> tensor;
      weight_buff->reserve(weight_dims, &tensor);
      weights_.push_back(tensor);
    }
    {
      Tensor2<float> tensor;
      wgrad_buff->reserve(weight_dims, &tensor);
      wgrad_.push_back(tensor);
    }

    blob_buff->reserve(out_dims, &wgrad_tmp_trans_);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

std::unique_ptr<DataSimulator> MultiplyLayer::get_uniform_initializer(const int index) {
  float bottom_dim = slot_num_;
  float top_dim = slot_num_ * embedding_vec_size_;

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> MultiplyLayer::get_xavier_uniform_initializer(const int index) {
  float bottom_dim = slot_num_;
  float top_dim = slot_num_ * embedding_vec_size_;

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> MultiplyLayer::get_xavier_norm_initializer(const int index) {
  float bottom_dim = slot_num_;
  float top_dim = slot_num_ * embedding_vec_size_;

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> MultiplyLayer::get_default_initializer(const int index) {
  float bottom_dim = slot_num_;
  float top_dim = slot_num_ * embedding_vec_size_;

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

void MultiplyLayer::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  float* input = in_tensors_[0].get_ptr();
  float* weight = weights_[0].get_ptr();
  float* output = out_tensors_[0].get_ptr();

  dim3 blockSize(embedding_vec_size_, 1, 1);
  dim3 gridSize(batch_size_, 1, 1);
  multiply_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
      input, weight, output, batch_size_, slot_num_, embedding_vec_size_);
}

void MultiplyLayer::bprop() {
  CudaDeviceContext context(get_device_id());

  float* weight = weights_[0].get_ptr();
  float* wgrad = wgrad_[0].get_ptr();
  float* wgrad_tmp_trans = wgrad_tmp_trans_.get_ptr();
  float* input = in_tensors_[0].get_ptr();
  float* output = out_tensors_[0].get_ptr();

  multiply_wgrad(output, input, wgrad, wgrad_tmp_trans, batch_size_, slot_num_, embedding_vec_size_,
                 get_gpu().get_stream());

  // CAUSION: dgrad computation will modify the "input", so it must be put after wgrad computation
  multiply_dgrad(output, weight, input, batch_size_, slot_num_, embedding_vec_size_,
                 get_gpu().get_stream());
}

}  // namespace HugeCTR

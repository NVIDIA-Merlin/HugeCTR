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

#include "HugeCTR/include/layers/fused_fully_connected_layer.hpp"
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

namespace {

__global__ void add_bias_and_re_kernel(__half* top, __half* middle, const __half* bias, int n,
                                       int ldn) {
  const __half2 zero = TypeFunc<__half2>::zero();
  __half2* top2 = reinterpret_cast<__half2*>(top);
  __half2* middle2 = reinterpret_cast<__half2*>(middle);
  const __half2* bias2 = reinterpret_cast<const __half2*>(bias);

  int offset = blockIdx.x * ldn;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    __half2 t = __hadd2(middle2[offset + tid], __ldg(bias2 + tid));
    middle2[offset + tid] = t;
    __half2 mask = __hgt2(t, zero);
    top2[offset + tid] = __hmul2(t, mask);
  }
}

template <int BLOCK_WIDTH>
__global__ void reverse_add_bias_and_re_kernel(float* bias, __half* middle, const __half* top,
                                               int ldn) {
  __shared__ __half2 elem[32][BLOCK_WIDTH + 1];
  __shared__ __half2 accu[BLOCK_WIDTH];

  const __half2 zero = TypeFunc<__half2>::zero();

  __half2* middle2 = reinterpret_cast<__half2*>(middle);
  const __half2* top2 = reinterpret_cast<const __half2*>(top);

  int lx, ly, gi;
  int gx_offset = blockIdx.x * BLOCK_WIDTH;
  int gy_offset = blockIdx.y * 32;

  for (int i = 0; i < BLOCK_WIDTH * 32; i += blockDim.x) {
    lx = threadIdx.x % BLOCK_WIDTH;
    ly = (i + threadIdx.x) / BLOCK_WIDTH;
    gi = (ly + gy_offset) * ldn + (lx + gx_offset);

    __half2 t = middle2[gi];
    __half2 mask = __hgt2(t, zero);
    t = __hmul2(__ldg(top2 + gi), mask);

    middle2[gi] = t;
    elem[ly][lx] = t;
  }

  __syncthreads();

  for (int i = 0; i < BLOCK_WIDTH * 32; i += blockDim.x) {
    lx = (i + threadIdx.x) / 32;
    ly = threadIdx.x % 32;

    __half2 val = warpReduceSum(elem[ly][lx]);
    if (ly == 0) {
      accu[lx] = val;
    }
  }

  __syncthreads();

  if (threadIdx.x < BLOCK_WIDTH * 2) {
    __half2 val = accu[threadIdx.x / 2];
    float fval = (threadIdx.x % 2 == 0) ? __low2float(val) : __high2float(val);
    atomicAdd(bias + gx_offset * 2 + threadIdx.x, fval);
  }
}

}  // namespace

FusedFullyConnectedLayer::FusedFullyConnectedLayer(
    const GeneralBufferPtr<float>& master_weights_buff,
    const GeneralBufferPtr<__half>& weights_buff, const GeneralBufferPtr<__half>& weights_grad_buff,
    const GeneralBufferPtr<float>& blobs_buff, const GeneralBufferPtr<__half>& blobs_half_buff,
    const TensorPtr<__half>& bottom_tensor, const TensorPtr<__half>& top_tensor,
    TensorFormat_t weight_tensor_format, cublasHandle_t const& cublas_handle, int device_id)
    : Layer(device_id),
      cublas_handle_(cublas_handle),
      falgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_x_(CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
  const auto& bottom_tensor_dim = bottom_tensor->get_dims();
  const auto& top_tensor_dim = top_tensor->get_dims();

  if (bottom_tensor_dim.size() != 2 || top_tensor_dim.size() != 2) {
    CK_THROW_(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
  }

  assert(weight_tensor_format == TensorFormat_t::HW);
  assert(bottom_tensor->get_format() == TensorFormat_t::HW);
  assert(top_tensor->get_format() == TensorFormat_t::HW);

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  if (m % 32 != 0 || n % 64 != 0) {
    CK_THROW_(Error_t::WrongInput,
              "The first dimension of bottom tensor must be a multiple of 32, the second dimension "
              "of top tensor must be a multiple of 64.");
  }

  std::vector<size_t> kernel_dim = {k, n};
  std::vector<size_t> bias_dim = {1, n};

  master_weights_.emplace_back(
      new Tensor<float>(kernel_dim, master_weights_buff, weight_tensor_format));
  master_weights_.emplace_back(
      new Tensor<float>(bias_dim, master_weights_buff, weight_tensor_format));

  weights_.emplace_back(new Tensor<__half>(kernel_dim, weights_buff, weight_tensor_format));
  weights_.emplace_back(new Tensor<__half>(bias_dim, weights_buff, weight_tensor_format));

  weights_grad_.emplace_back(
      new Tensor<__half>(kernel_dim, weights_grad_buff, weight_tensor_format));
  weights_grad_.emplace_back(new Tensor<__half>(bias_dim, weights_grad_buff, weight_tensor_format));

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
  middle_tensor_.reset(
      new Tensor<__half>(top_tensor_->get_dims(), blobs_half_buff, TensorFormat_t::HW));
  bias_grad_tensor_.reset(new Tensor<float>(bias_dim, blobs_buff, TensorFormat_t::HW));
}

void FusedFullyConnectedLayer::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  CK_CUBLAS_THROW_(cublasSetStream(cublas_handle_, stream));

  const __half* kernel = weights_[0]->get_ptr();
  const __half* bias = weights_[1]->get_ptr();
  const __half* bottom = bottom_tensor_->get_ptr();
  __half* middle = middle_tensor_->get_ptr();
  __half* top = top_tensor_->get_ptr();

  const auto& bottom_tensor_dim = bottom_tensor_->get_dims();
  const auto& top_tensor_dim = top_tensor_->get_dims();

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, kernel,
                                CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta, middle, CUDA_R_16F, n,
                                CUDA_R_32F, falgo_k_));

  const size_t max_threads = 1024;
  const size_t blocks = m;
  const size_t threads = min(n / 2, max_threads);
  add_bias_and_re_kernel<<<blocks, threads, 0, stream>>>(top, middle, bias, n / 2, n / 2);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

void FusedFullyConnectedLayer::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  CK_CUBLAS_THROW_(cublasSetStream(cublas_handle_, stream));

  const __half* kernel = weights_[0]->get_ptr();
  const __half* top = top_tensor_->get_ptr();
  __half* kernel_grad = weights_grad_[0]->get_ptr();
  __half* bias_grad = weights_grad_[1]->get_ptr();
  __half* bottom = bottom_tensor_->get_ptr();
  __half* middle = middle_tensor_->get_ptr();
  float* bias_grad_float = bias_grad_tensor_->get_ptr();

  const auto& bottom_tensor_dim = bottom_tensor_->get_dims();
  const auto& top_tensor_dim = top_tensor_->get_dims();

  int m = bottom_tensor_dim[0];
  int n = top_tensor_dim[1];
  int k = bottom_tensor_dim[1];

  const float alpha = 1.0f;
  const float beta_k = 1.0f;
  const float beta_x = 0.0f;

  initialize_array<<<(n - 1) / 1024 + 1, 1024, 0, stream>>>(bias_grad_float, n, 0.0f);

  dim3 blocks(n / 64, m / 32);
  reverse_add_bias_and_re_kernel<32>
      <<<blocks, 512, 0, stream>>>(bias_grad_float, middle, top, n / 2);

  convert_array<<<(n - 1) / 1024 + 1, 1024, 0, stream>>>(bias_grad, bias_grad_float, n);

  CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, middle,
                                CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta_k, kernel_grad,
                                CUDA_R_16F, n, CUDA_R_32F, balgo_k_));

  CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, kernel,
                                CUDA_R_16F, n, middle, CUDA_R_16F, n, &beta_x, bottom, CUDA_R_16F,
                                k, CUDA_R_32F, balgo_x_));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

void FusedFullyConnectedLayer::optimize() {
  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());
  const size_t repeat_num = 100;

  // CUDA stream to be used for cublas on this device
  cudaStream_t stream;
  CK_CUDA_THROW_(cudaStreamCreate(&stream));

  // Set stream to cublas handler
  CK_CUBLAS_THROW_(cublasSetStream(cublas_handle_, stream));

  // Device Tensors to be used
  __half* bottom = bottom_tensor_->get_ptr();
  __half* top = top_tensor_->get_ptr();
  __half* kernel = weights_[0]->get_ptr();
  __half* bias = weights_[1]->get_ptr();
  __half* kernel_grad = weights_grad_[0]->get_ptr();
  __half* bias_grad = weights_grad_[1]->get_ptr();

  // Tensor dim
  const auto& bottom_tensor_dim = bottom_tensor_->get_dims();
  const auto& top_tensor_dim = top_tensor_->get_dims();

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  // Record time for each algorithm
  float shortestTime = std::numeric_limits<float>::max();
  float time;
  cudaEvent_t start, stop;
  CK_CUDA_THROW_(cudaEventCreate(&start));
  CK_CUDA_THROW_(cudaEventCreate(&stop));

  // Start, end for search
  const cublasGemmAlgo_t startAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  const cublasGemmAlgo_t endAlgo = CUBLAS_GEMM_ALGO15_TENSOR_OP;

  // Search all the algorithm for falgo_k_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, stream));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, kernel,
                            CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta, top, CUDA_R_16F, n,
                            CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, stream));
    CK_CUDA_THROW_(cudaEventSynchronize(stop));
    CK_CUDA_THROW_(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      printf("The algorithms %d is not supported for fprop, skipped.\n", testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      falgo_k_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

  // Search all the algorithm for balgo_k_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, stream));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, top,
                            CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta, kernel_grad, CUDA_R_16F, n,
                            CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, stream));
    CK_CUDA_THROW_(cudaEventSynchronize(stop));
    CK_CUDA_THROW_(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      printf("The algorithms %d is not supported for bprop_W, skipped.\n", testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_k_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

  // Search all the algorithm for balgo_x_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, stream));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, kernel,
                            CUDA_R_16F, n, top, CUDA_R_16F, n, &beta, bottom, CUDA_R_16F, k,
                            CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }

    CK_CUDA_THROW_(cudaEventRecord(stop, stream));
    CK_CUDA_THROW_(cudaEventSynchronize(stop));
    CK_CUDA_THROW_(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      printf("The algorithms %d is not supported for bprop_Xn, skipped.\n", testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_x_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Print selection information
  // printf("The algorithm selection for falgo_k_, balgo_k_, balgo_x_ are: %d, %d and %d.\n",
  //        (int)falgo_k_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP,
  //        (int)balgo_k_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP,
  //        (int)balgo_x_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // Output msg
  //MESSAGE_("The fully-connected layer has finished choosing the algorithm for cublas Gemm.");
  // Clean-up
  CK_CUDA_THROW_(cudaEventDestroy(start));
  CK_CUDA_THROW_(cudaEventDestroy(stop));
  CK_CUDA_THROW_(cudaStreamDestroy(stream));
}  // namespace HugeCTR

std::vector<float> FusedFullyConnectedLayer::get_initializer() {
  const size_t kernel_elements = weights_[0]->get_num_elements();
  const size_t bias_elements = weights_[1]->get_num_elements();
  size_t bottom_dim = bottom_tensor_->get_dims()[1];
  size_t top_dim = top_tensor_->get_dims()[1];

  std::vector<float> buffer(kernel_elements + bias_elements, 0.0f);

  {
    float stddev = sqrtf(2.0f / (bottom_dim + top_dim));
    HugeCTR::GaussianDataSimulator<float> simulator(0, stddev, -10 * stddev, 10 * stddev);
    for (size_t i = 0; i < kernel_elements; i++) {
      buffer[i] = simulator.get_num();
    }
  }

  {
    float stddev = sqrtf(1.0f / top_dim);
    HugeCTR::GaussianDataSimulator<float> simulator(0, stddev, -10 * stddev, 10 * stddev);
    for (size_t i = 0; i < bias_elements; i++) {
      buffer[i + kernel_elements] = simulator.get_num();
    }
  }

  return buffer;
}

}  // namespace HugeCTR

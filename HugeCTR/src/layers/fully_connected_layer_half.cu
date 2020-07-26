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

#include "HugeCTR/include/layers/fully_connected_layer_half.hpp"
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

FullyConnectedLayerHalf::FullyConnectedLayerHalf(const GeneralBufferPtr<float>& master_weights_buff,
                                                 const GeneralBufferPtr<__half>& weights_buff,
                                                 const GeneralBufferPtr<__half>& weights_grad_buff,
                                                 const GeneralBufferPtr<__half>& blobs_buff,
                                                 const TensorPtr<__half>& bottom_tensor,
                                                 const TensorPtr<__half>& top_tensor,
                                                 TensorFormat_t weight_tensor_format,
                                                 cublasHandle_t const& cublas_handle, int device_id,
                                                 std::vector<Initializer_t> initializer_types)
    : Layer(device_id, initializer_types),
      cublas_handle_(cublas_handle),
      falgo_b_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      falgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_b_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
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

  std::vector<size_t> kernel_dim = {k, n};
  std::vector<size_t> bias_dim = {1, n};
  std::vector<size_t> identity_dim = {1, m};

  weights_.emplace_back(new Tensor<float>(kernel_dim, master_weights_buff, weight_tensor_format));
  weights_.emplace_back(new Tensor<float>(bias_dim, master_weights_buff, weight_tensor_format));

  weights_half_.emplace_back(new Tensor<__half>(kernel_dim, weights_buff, weight_tensor_format));
  weights_half_.emplace_back(new Tensor<__half>(bias_dim, weights_buff, weight_tensor_format));

  weights_grad_.emplace_back(
      new Tensor<__half>(kernel_dim, weights_grad_buff, weight_tensor_format));
  weights_grad_.emplace_back(new Tensor<__half>(bias_dim, weights_grad_buff, weight_tensor_format));

  identity_tensor_.reset(new Tensor<__half>(identity_dim, blobs_buff, TensorFormat_t::HW));

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
}

void FullyConnectedLayerHalf::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  CK_CUBLAS_THROW_(cublasSetStream(cublas_handle_, stream));

  const __half* kernel = weights_half_[0]->get_ptr();
  const __half* bias = weights_half_[1]->get_ptr();
  const __half* bottom = bottom_tensor_->get_ptr();
  const __half* identity = identity_tensor_->get_ptr();
  __half* top = top_tensor_->get_ptr();

  const auto& bottom_tensor_dim = bottom_tensor_->get_dims();
  const auto& top_tensor_dim = top_tensor_->get_dims();

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  const float alpha = 1.0f;
  const float beta_b = 0.0f;
  const float beta_k = 1.0f;

  CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &alpha, bias,
                                CUDA_R_16F, n, identity, CUDA_R_16F, 1, &beta_b, top, CUDA_R_16F, n,
                                CUDA_R_32F, falgo_b_));

  CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, kernel,
                                CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta_k, top, CUDA_R_16F, n,
                                CUDA_R_32F, falgo_k_));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

void FullyConnectedLayerHalf::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  CK_CUBLAS_THROW_(cublasSetStream(cublas_handle_, stream));

  const __half* kernel = weights_half_[0]->get_ptr();
  const __half* top = top_tensor_->get_ptr();
  const __half* identity = identity_tensor_->get_ptr();
  __half* kernel_grad = weights_grad_[0]->get_ptr();
  __half* bias_grad = weights_grad_[1]->get_ptr();
  __half* bottom = bottom_tensor_->get_ptr();

  const auto& bottom_tensor_dim = bottom_tensor_->get_dims();
  const auto& top_tensor_dim = top_tensor_->get_dims();

  int m = bottom_tensor_dim[0];
  int n = top_tensor_dim[1];
  int k = bottom_tensor_dim[1];

  const float alpha = 1.0f;
  const float beta_b = 0.0f;
  const float beta_k = 1.0f;
  const float beta_x = 0.0f;

  CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, m, &alpha, top,
                                CUDA_R_16F, n, identity, CUDA_R_16F, m, &beta_b, bias_grad,
                                CUDA_R_16F, n, CUDA_R_32F, balgo_b_));

  CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, top,
                                CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta_k, kernel_grad,
                                CUDA_R_16F, n, CUDA_R_32F, balgo_k_));

  CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, kernel,
                                CUDA_R_16F, n, top, CUDA_R_16F, n, &beta_x, bottom, CUDA_R_16F, k,
                                CUDA_R_32F, balgo_x_));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

void FullyConnectedLayerHalf::initialize() {
  CudaDeviceContext context(get_device_id());

  // CUDA stream to be used for cublas on this device
  cudaStream_t stream;
  CK_CUDA_THROW_(cudaStreamCreate(&stream));

  __half* identity = identity_tensor_->get_ptr();
  const auto& bottom_tensor_dim = bottom_tensor_->get_dims();
  size_t m = bottom_tensor_dim[0];

  // Initialize identity vector
  initialize_array<<<(m - 1) / 1024 + 1, 1024, 0, stream>>>(identity, m, __float2half(1.0f));

  CK_CUDA_THROW_(cudaStreamDestroy(stream));
}

void FullyConnectedLayerHalf::search_algorithm() {
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
  __half* identity = identity_tensor_->get_ptr();
  __half* kernel = weights_half_[0]->get_ptr();
  __half* bias = weights_half_[1]->get_ptr();
  __half* kernel_grad = weights_grad_[0]->get_ptr();
  __half* bias_grad = weights_grad_[1]->get_ptr();

  // Tensor dim
  const auto& bottom_tensor_dim = bottom_tensor_->get_dims();
  const auto& top_tensor_dim = top_tensor_->get_dims();

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  // Initialize identity vector
  initialize_array<<<(m - 1) / 1024 + 1, 1024, 0, stream>>>(identity, m, __float2half(1.0f));

  // Record time for each algorithm
  float shortestTime = std::numeric_limits<float>::max();
  float time;
  cudaEvent_t start, stop;
  CK_CUDA_THROW_(cudaEventCreate(&start));
  CK_CUDA_THROW_(cudaEventCreate(&stop));

  // Start, end for search
  const cublasGemmAlgo_t startAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  const cublasGemmAlgo_t endAlgo = CUBLAS_GEMM_ALGO15_TENSOR_OP;

  // Search all the algorithm for falgo_b_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, stream));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &alpha, bias,
                            CUDA_R_16F, n, identity, CUDA_R_16F, 1, &beta, top, CUDA_R_16F, n,
                            CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, stream));
    CK_CUDA_THROW_(cudaEventSynchronize(stop));
    CK_CUDA_THROW_(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      // printf("The algorithms %d is not supported for fprop_b, skipped.\n", testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      falgo_b_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

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
      // printf("The algorithms %d is not supported for fprop, skipped.\n", testAlgo);
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

  // Search all the algorithm for balgo_b_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, stream));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, m, &alpha, top,
                            CUDA_R_16F, n, identity, CUDA_R_16F, m, &beta, bias_grad, CUDA_R_16F, n,
                            CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, stream));
    CK_CUDA_THROW_(cudaEventSynchronize(stop));
    CK_CUDA_THROW_(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      // printf("The algorithms %d is not supported for bprop_W, skipped.\n", testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_b_ = static_cast<cublasGemmAlgo_t>(testAlgo);
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
      // printf("The algorithms %d is not supported for bprop_W, skipped.\n", testAlgo);
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
      // printf("The algorithms %d is not supported for bprop_Xn, skipped.\n", testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_x_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Print selection information
  // printf(
  //     "The algorithm selection for falgo_b_, falgo_k_, balgo_b_, balgo_k_, balgo_x_ are: %d, %d,
  //     "
  //     "%d, %d and %d.\n",
  //     (int)falgo_b_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP, (int)falgo_k_ -
  //     CUBLAS_GEMM_DEFAULT_TENSOR_OP, (int)balgo_b_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP, (int)balgo_k_
  //     - CUBLAS_GEMM_DEFAULT_TENSOR_OP, (int)balgo_x_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // Output msg
  // MESSAGE_("The fully-connected layer has finished choosing the algorithm for cublas Gemm.");
  // Clean-up
  CK_CUDA_THROW_(cudaEventDestroy(start));
  CK_CUDA_THROW_(cudaEventDestroy(stop));
  CK_CUDA_THROW_(cudaStreamDestroy(stream));
}  // namespace HugeCTR

std::unique_ptr<DataSimulator<float>> FullyConnectedLayerHalf::get_uniform_initializer(
    const int index) {
  size_t bottom_dim = bottom_tensor_->get_dims()[1];
  size_t top_dim = top_tensor_->get_dims()[1];

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::unique_ptr<DataSimulator<float>>(new UnifiedDataSimulator<float>(-1 * limit, limit));
}

std::unique_ptr<DataSimulator<float>> FullyConnectedLayerHalf::get_xavier_uniform_initializer(
    const int index) {
  size_t bottom_dim = bottom_tensor_->get_dims()[1];
  size_t top_dim = top_tensor_->get_dims()[1];

  return std::unique_ptr<DataSimulator<float>>(new VarianceScalingSimulator<float>(
      1.f, data_simu::Mode_t::Fan_avg, data_simu::Distribution_t::Uniform,
      0 == index ? bottom_dim : 0, top_dim));
}

std::unique_ptr<DataSimulator<float>> FullyConnectedLayerHalf::get_xavier_norm_initializer(
    const int index) {
  size_t bottom_dim = bottom_tensor_->get_dims()[1];
  size_t top_dim = top_tensor_->get_dims()[1];

  return std::unique_ptr<DataSimulator<float>>(new VarianceScalingSimulator<float>(
      1.f, data_simu::Mode_t::Fan_avg, data_simu::Distribution_t::Norm, 0 == index ? bottom_dim : 0,
      top_dim));
}

std::unique_ptr<DataSimulator<float>> FullyConnectedLayerHalf::get_default_initializer(
    const int index) {
  size_t bottom_dim = bottom_tensor_->get_dims()[1];
  size_t top_dim = top_tensor_->get_dims()[1];

  std::unique_ptr<DataSimulator<float>> simu(nullptr);
  if (0 == index) {
    simu.reset(new VarianceScalingSimulator<float>(
        1.f, data_simu::Mode_t::Fan_avg, data_simu::Distribution_t::Norm, bottom_dim, top_dim));
  } else if (1 == index) {
    float stddev = sqrt(1.f / top_dim);
    simu.reset(new GaussianDataSimulator<float>(0, stddev, -2 * stddev, 2 * stddev));
  } else {
    CK_THROW_(Error_t::OutOfBound, "index != {0, 1}.");
  }

  return simu;
}

}  // namespace HugeCTR

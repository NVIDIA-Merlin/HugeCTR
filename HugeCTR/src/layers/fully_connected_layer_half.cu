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

#include <layers/fully_connected_layer_half.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

FullyConnectedLayer<__half>::FullyConnectedLayer(
    const std::shared_ptr<BufferBlock2<float>>& master_weights_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weights_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weights_grad_buff,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const Tensor2<__half>& bottom_tensor, const Tensor2<__half>& top_tensor,
    const std::shared_ptr<GPUResource>& gpu_resource, std::vector<Initializer_t> initializer_types)
    : Layer(gpu_resource, initializer_types),
      falgo_b_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      falgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_b_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_x_(CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
  const auto& bottom_tensor_dim = bottom_tensor.get_dimensions();
  const auto& top_tensor_dim = top_tensor.get_dimensions();

  if (bottom_tensor_dim.size() != top_tensor_dim.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "input or output tensor don't have same dimensions");
  }
  size_t in_batch = 1;
  size_t out_batch = 1;
  size_t in_hidden_dim = bottom_tensor_dim[bottom_tensor_dim.size() - 1];
  size_t out_hidden_dim = top_tensor_dim[top_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < bottom_tensor_dim.size() - 1; idx++) {
    in_batch = in_batch * bottom_tensor_dim[idx];
    out_batch = out_batch * top_tensor_dim[idx];
  }

  size_t m = in_batch;
  size_t n = out_hidden_dim;
  size_t k = in_hidden_dim;
  if (in_batch != out_batch) {
    HCTR_OWN_THROW(Error_t::WrongInput, "size of input / output tensor doesn't match");
  }

  std::vector<size_t> kernel_dim = {k, n};
  std::vector<size_t> bias_dim = {1, n};
  std::vector<size_t> identity_dim = {1, m};

  {
    Tensor2<float> tensor;
    master_weights_buff->reserve(kernel_dim, &tensor);
    weights_.push_back(tensor);
  }
  {
    Tensor2<float> tensor;
    master_weights_buff->reserve(bias_dim, &tensor);
    weights_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_buff->reserve(kernel_dim, &tensor);
    weights_half_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_buff->reserve(bias_dim, &tensor);
    weights_half_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_grad_buff->reserve(kernel_dim, &tensor);
    weights_grad_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_grad_buff->reserve(bias_dim, &tensor);
    weights_grad_.push_back(tensor);
  }
  blobs_buff->reserve(identity_dim, &identity_tensor_);

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
}

void FullyConnectedLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = weights_half_[0].get_ptr();
  const __half* bias = weights_half_[1].get_ptr();
  const __half* bottom = get_bottom_tensor(is_train).get_ptr();
  const __half* identity = identity_tensor_.get_ptr();
  __half* top = top_tensor_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor(is_train).get_dimensions();
  const auto& top_tensor_dim = top_tensor_.get_dimensions();

  size_t in_batch = 1;
  size_t in_hidden_dim = bottom_tensor_dim[bottom_tensor_dim.size() - 1];
  size_t out_hidden_dim = top_tensor_dim[top_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < bottom_tensor_dim.size() - 1; idx++) {
    in_batch = in_batch * bottom_tensor_dim[idx];
  }
  size_t m = in_batch;
  size_t n = out_hidden_dim;
  size_t k = in_hidden_dim;

  const float alpha = 1.0f;
  const float beta_b = 0.0f;
  const float beta_k = 1.0f;

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1,
                              &alpha, bias, CUDA_R_16F, n, identity, CUDA_R_16F, 1, &beta_b, top,
                              CUDA_R_16F, n, CUDA_R_32F, falgo_b_));

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                              &alpha, kernel, CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta_k, top,
                              CUDA_R_16F, n, CUDA_R_32F, falgo_k_));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

void FullyConnectedLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = weights_half_[0].get_ptr();
  const __half* top = top_tensor_.get_ptr();
  const __half* identity = identity_tensor_.get_ptr();
  __half* kernel_grad = weights_grad_[0].get_ptr();
  __half* bias_grad = weights_grad_[1].get_ptr();
  __half* bottom = get_bottom_tensor(true).get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor(true).get_dimensions();
  const auto& top_tensor_dim = top_tensor_.get_dimensions();

  size_t in_batch = 1;
  size_t in_hidden_dim = bottom_tensor_dim[bottom_tensor_dim.size() - 1];
  size_t out_hidden_dim = top_tensor_dim[top_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < bottom_tensor_dim.size() - 1; idx++) {
    in_batch = in_batch * bottom_tensor_dim[idx];
  }

  size_t m = in_batch;
  size_t n = out_hidden_dim;
  size_t k = in_hidden_dim;

  const float alpha = 1.0f;
  const float beta_b = 0.0f;
  const float beta_k = 1.0f;
  const float beta_x = 0.0f;

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, 1, m,
                              &alpha, top, CUDA_R_16F, n, identity, CUDA_R_16F, m, &beta_b,
                              bias_grad, CUDA_R_16F, n, CUDA_R_32F, balgo_b_));

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
                              &alpha, top, CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta_k,
                              kernel_grad, CUDA_R_16F, n, CUDA_R_32F, balgo_k_));

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, k, m, n,
                              &alpha, kernel, CUDA_R_16F, n, top, CUDA_R_16F, n, &beta_x, bottom,
                              CUDA_R_16F, k, CUDA_R_32F, balgo_x_));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

void FullyConnectedLayer<__half>::initialize() {
  CudaDeviceContext context(get_device_id());

  __half* identity = identity_tensor_.get_ptr();
  const auto& bottom_tensor_dim = get_bottom_tensor(true).get_dimensions();
  size_t m = 1;
  for (size_t idx = 0; idx < bottom_tensor_dim.size() - 1; idx++) {
    m = m * bottom_tensor_dim[idx];
  }
  // Initialize identity vector
  initialize_array<<<(m - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(identity, m,
                                                                            __float2half(1.0f));
}

void FullyConnectedLayer<__half>::search_algorithm() {
  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());

  const size_t repeat_num = 100;

  // Device Tensors to be used
  __half* bottom = get_bottom_tensor(true).get_ptr();
  __half* top = top_tensor_.get_ptr();
  __half* identity = identity_tensor_.get_ptr();
  __half* kernel = weights_half_[0].get_ptr();
  __half* bias = weights_half_[1].get_ptr();
  __half* kernel_grad = weights_grad_[0].get_ptr();
  __half* bias_grad = weights_grad_[1].get_ptr();

  // Tensor dim
  const auto& bottom_tensor_dim = get_bottom_tensor(true).get_dimensions();
  const auto& top_tensor_dim = top_tensor_.get_dimensions();

  size_t in_batch = 1;
  size_t in_hidden_dim = bottom_tensor_dim[bottom_tensor_dim.size() - 1];
  size_t out_hidden_dim = top_tensor_dim[top_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < bottom_tensor_dim.size() - 1; idx++) {
    in_batch = in_batch * bottom_tensor_dim[idx];
  }

  size_t m = in_batch;
  size_t n = out_hidden_dim;
  size_t k = in_hidden_dim;

  // Record time for each algorithm
  float shortestTime = std::numeric_limits<float>::max();
  float time;
  cudaEvent_t start, stop;
  HCTR_LIB_THROW(cudaEventCreate(&start));
  HCTR_LIB_THROW(cudaEventCreate(&stop));

  // Start, end for search
  const cublasGemmAlgo_t startAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  const cublasGemmAlgo_t endAlgo = CUBLAS_GEMM_ALGO15_TENSOR_OP;

  // Search all the algorithm for falgo_b_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Record start event
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1,
                            &alpha, bias, CUDA_R_16F, n, identity, CUDA_R_16F, 1, &beta, top,
                            CUDA_R_16F, n, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      // HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for fprop_b, skipped.\n",
      // testAlgo);
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
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, kernel, CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta, top,
                            CUDA_R_16F, n, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      // HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for fprop, skipped.\n",
      // testAlgo);
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
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, 1, m,
                            &alpha, top, CUDA_R_16F, n, identity, CUDA_R_16F, m, &beta, bias_grad,
                            CUDA_R_16F, n, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      // HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for bprop_W, skipped.\n",
      // testAlgo);
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
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
                            &alpha, top, CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta, kernel_grad,
                            CUDA_R_16F, n, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      // HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for bprop_W, skipped.\n",
      // testAlgo);
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
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, k, m, n,
                            &alpha, kernel, CUDA_R_16F, n, top, CUDA_R_16F, n, &beta, bottom,
                            CUDA_R_16F, k, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }

    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      // HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for bprop_Xn, skipped.\n",
      // testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_x_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Print selection information
  // HCTR_LOG(INFO, WORLD,
  //     "The algorithm selection for falgo_b_, falgo_k_, balgo_b_, balgo_k_, balgo_x_ are: %d, %d,
  //     "
  //     "%d, %d and %d.\n",
  //     (int)falgo_b_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP, (int)falgo_k_ -
  //     CUBLAS_GEMM_DEFAULT_TENSOR_OP, (int)balgo_b_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP, (int)balgo_k_
  //     - CUBLAS_GEMM_DEFAULT_TENSOR_OP, (int)balgo_x_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // Output msg
  // HCTR_LOG(INFO, ROOT, "The fully-connected layer has finished choosing the algorithm for cublas
  // Gemm.\n"); Clean-up
  HCTR_LIB_THROW(cudaEventDestroy(start));
  HCTR_LIB_THROW(cudaEventDestroy(stop));
}  // namespace HugeCTR

std::unique_ptr<DataSimulator> FullyConnectedLayer<__half>::get_uniform_initializer(
    const int index) {
  size_t bottom_dim =
      get_bottom_tensor(true).get_dimensions()[get_bottom_tensor(true).get_dimensions().size() - 1];
  size_t top_dim = top_tensor_.get_dimensions()[top_tensor_.get_dimensions().size() - 1];

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<__half>::get_xavier_uniform_initializer(
    const int index) {
  size_t bottom_dim =
      get_bottom_tensor(true).get_dimensions()[get_bottom_tensor(true).get_dimensions().size() - 1];
  size_t top_dim = top_tensor_.get_dimensions()[top_tensor_.get_dimensions().size() - 1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<__half>::get_xavier_norm_initializer(
    const int index) {
  size_t bottom_dim =
      get_bottom_tensor(true).get_dimensions()[get_bottom_tensor(true).get_dimensions().size() - 1];
  size_t top_dim = top_tensor_.get_dimensions()[top_tensor_.get_dimensions().size() - 1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<__half>::get_default_initializer(
    const int index) {
  size_t bottom_dim =
      get_bottom_tensor(true).get_dimensions()[get_bottom_tensor(true).get_dimensions().size() - 1];
  size_t top_dim = top_tensor_.get_dimensions()[top_tensor_.get_dimensions().size() - 1];

  std::unique_ptr<DataSimulator> simu(nullptr);
  if (0 == index) {
    simu.reset(new VarianceScalingSimulator(1.f, data_simu::Mode_t::Fan_avg,
                                            data_simu::Distribution_t::Norm, bottom_dim, top_dim));
  } else if (1 == index) {
    float stddev = sqrt(1.f / top_dim);
    simu.reset(new GaussianDataSimulator(0, stddev, -2 * stddev, 2 * stddev));
  } else {
    HCTR_OWN_THROW(Error_t::OutOfBound, "index != {0, 1}.");
  }

  return simu;
}

template class FullyConnectedLayer<__half>;

}  // namespace HugeCTR

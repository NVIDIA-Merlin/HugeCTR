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

#include <math.h>

#include <layers/fully_connected_layer.hpp>
#include <linalg/matrix_vector_op.cuh>
#include <linalg/reduce.cuh>
#include <utils.cuh>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

FullyConnectedLayer<float>::FullyConnectedLayer(
    const std::shared_ptr<BufferBlock2<float>>& weight_buff,
    const std::shared_ptr<BufferBlock2<float>>& wgrad_buff, const Tensor2<float>& in_tensor,
    const Tensor2<float>& out_tensor, const std::shared_ptr<GPUResource>& gpu_resource,
    bool use_mixed_precision, bool enable_tf32_compute,
    std::vector<Initializer_t> initializer_types)
    : Layer(gpu_resource, initializer_types),
      use_mixed_precision_(use_mixed_precision),
      enable_tf32_compute_(enable_tf32_compute) {
  try {
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor.get_dimensions();
    const auto& out_tensor_dim = out_tensor.get_dimensions();
    // 1. two dim?
    if (in_tensor_dim.size() != 2 || out_tensor_dim.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
    }
    // 2. dim match?
    size_t m = in_tensor_dim[0];
    size_t n = out_tensor_dim[1];
    size_t k = in_tensor_dim[1];
    size_t m_ck = out_tensor_dim[0];
    if (m != m_ck) {
      CK_THROW_(Error_t::WrongInput, "size of input / output tensor doesn't match");
    }

    std::vector<size_t> weight_dim = {k, n};
    std::vector<size_t> bias_dim = {1, n};

    {
      Tensor2<float> tensor;
      weight_buff->reserve(weight_dim, &tensor);
      weights_.push_back(tensor);
    }
    {
      Tensor2<float> tensor;
      weight_buff->reserve(bias_dim, &tensor);
      weights_.push_back(tensor);
    }
    {
      Tensor2<float> tensor;
      wgrad_buff->reserve(weight_dim, &tensor);
      wgrad_.push_back(tensor);
    }
    {
      Tensor2<float> tensor;
      wgrad_buff->reserve(bias_dim, &tensor);
      wgrad_.push_back(tensor);
    }
    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);
    // Where should we create this cuBLAS handle?
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void __global__ add_bias_kernel_row(float* data, const float* bias, const int m, const int n) {
  int offset = blockIdx.x * n;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    data[offset + tid] += bias[tid];
  }
}
void __global__ add_bias_kernel_col(float* data, const float* bias, const int m, const int n) {
  int offset = blockIdx.x * m;
  float b = bias[blockIdx.x];
  for (int tid = threadIdx.x; tid < m; tid += blockDim.x) {
    data[offset + tid] += b;
  }
}
void add_bias(float* data, const float* bias, const int m, const int n, bool row_major,
              cudaStream_t stream) {
  if (row_major) {
    dim3 grid(m);
    dim3 block(min(n, 1024));
    add_bias_kernel_row<<<grid, block, 0, stream>>>(data, bias, m, n);
  } else {
    dim3 grid(n);
    dim3 block(min(m, 1024));
    add_bias_kernel_col<<<grid, block, 0, stream>>>(data, bias, m, n);
  }
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

void FullyConnectedLayer<float>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  Tensor2<float>& in_tensor = get_in_tensors(is_train)[0];
  Tensor2<float>& out_tensor = out_tensors_[0];

  float* weight = weights_[0].get_ptr();
  float* bias = weights_[1].get_ptr();
  float* in = in_tensor.get_ptr();
  float* out = out_tensor.get_ptr();

  const auto& in_tensor_dim = in_tensor.get_dimensions();
  const auto& out_tensor_dim = out_tensor.get_dimensions();

  int m, n, k;

  m = in_tensor_dim[0];
  n = out_tensor_dim[1];
  k = in_tensor_dim[1];

  float alpha = 1.0f, beta = 0.0f;

  cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  CK_CUBLAS_THROW_(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                &alpha, weight, CUDA_R_32F, n, in, CUDA_R_32F, k, &beta, out,
                                CUDA_R_32F, n, compute_type, falgo_));
  add_bias(out, bias, m, n, true, get_gpu().get_stream());
}

void FullyConnectedLayer<float>::bprop() {
  CudaDeviceContext context(get_device_id());

  Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  Tensor2<float>& out_tensor = out_tensors_[0];

  float* wgrad = wgrad_[0].get_ptr();
  float* bias_grad = wgrad_[1].get_ptr();
  float* weight = weights_[0].get_ptr();
  float* in = in_tensor.get_ptr();
  float* out = out_tensor.get_ptr();

  const auto& in_tensor_dim = in_tensor.get_dimensions();
  const auto& out_tensor_dim = out_tensor.get_dimensions();

  int m, n, k;

  m = in_tensor_dim[0];
  n = out_tensor_dim[1];
  k = in_tensor_dim[1];

  float alpha = 1.0f, beta_w = 1.0f, beta_x = 0.0f;

  cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  // gradient respect to W
  CK_CUBLAS_THROW_(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
                                &alpha, out, CUDA_R_32F, n, in, CUDA_R_32F, k, &beta_w, wgrad,
                                CUDA_R_32F, n, compute_type, balgo_W_));
  // gradient respect to Xn
  CK_CUBLAS_THROW_(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, k, m, n,
                                &alpha, weight, CUDA_R_32F, n, out, CUDA_R_32F, n, &beta_x, in,
                                CUDA_R_32F, k, compute_type, balgo_Xn_));
  MLCommon::LinAlg::reduce(bias_grad, out, m, n, float(0), false, true, get_gpu().get_stream(),
                           true);
}

void FullyConnectedLayer<float>::search_algorithm() {
  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());

  const int repeat_num = 100;

  // Device Tensors to be used
  Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  Tensor2<float>& out_tensor = out_tensors_[0];
  float* weight = weights_[0].get_ptr();
  float* in = in_tensor.get_ptr();
  float* out = out_tensor.get_ptr();
  float* wgrad = wgrad_[0].get_ptr();

  // Tensor dim
  const auto& in_tensor_dim = in_tensor.get_dimensions();
  const auto& out_tensor_dim = out_tensor.get_dimensions();

  int m, n, k;
  m = in_tensor_dim[0];
  n = out_tensor_dim[1];
  k = in_tensor_dim[1];

  // Record time for each algorithm
  float shortestTime = 100000000.0;
  float time;
  cudaEvent_t start, stop;
  CK_CUDA_THROW_(cudaEventCreate(&start));
  CK_CUDA_THROW_(cudaEventCreate(&stop));

  // cublas ret status
  cublasStatus_t status;

  // Start, end for search
  int startAlgo, endAlgo;
  if (use_mixed_precision_) {
    startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
  } else {
    startAlgo = (int)CUBLAS_GEMM_DEFAULT;
    endAlgo = (int)CUBLAS_GEMM_ALGO23;
  }

  cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  // Search all the algorithm for fprop
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta = 0.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (int i = 0; i < repeat_num; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, weight, CUDA_R_32F, n, in, CUDA_R_32F, k, &beta, out,
                            CUDA_R_32F, n, compute_type, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, get_gpu().get_stream()));
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
      falgo_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = 100000000.0;

  // Search all the algorithm for bprop_W
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta_w = 1.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (int i = 0; i < repeat_num; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
                            &alpha, out, CUDA_R_32F, n, in, CUDA_R_32F, k, &beta_w, wgrad,
                            CUDA_R_32F, n, compute_type, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, get_gpu().get_stream()));
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
      balgo_W_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = 100000000.0;

  // Search all the algorithm for bprop_Xn
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta_x = 0.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (int i = 0; i < repeat_num; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, k, m, n,
                            &alpha, weight, CUDA_R_32F, n, out, CUDA_R_32F, n, &beta_x, in,
                            CUDA_R_32F, k, compute_type, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, get_gpu().get_stream()));
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
      balgo_Xn_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Print selection information
  // printf("The algorithm selection for fprop, bprop_W and bprop_Xn are: %d, %d and %d.\n",
  //       (int)falgo_, (int)balgo_W_, (int)balgo_Xn_);

  // Output msg
  // MESSAGE_("The fully-connected layer has finished choosing the algorithm for cublas Gemm.");
  // Clean-up
  CK_CUDA_THROW_(cudaEventDestroy(start));
  CK_CUDA_THROW_(cudaEventDestroy(stop));
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<float>::get_uniform_initializer(
    const int index) {
  const Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  const Tensor2<float>& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor.get_dimensions()[1];
  float top_dim = out_tensor.get_dimensions()[1];

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<float>::get_xavier_uniform_initializer(
    const int index) {
  const Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  const Tensor2<float>& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor.get_dimensions()[1];
  float top_dim = out_tensor.get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<float>::get_xavier_norm_initializer(
    const int index) {
  const Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  const Tensor2<float>& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor.get_dimensions()[1];
  float top_dim = out_tensor.get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<float>::get_default_initializer(
    const int index) {
  const Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  const Tensor2<float>& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor.get_dimensions()[1];
  float top_dim = out_tensor.get_dimensions()[1];

  std::unique_ptr<DataSimulator> simu(nullptr);
  if (0 == index) {
    simu.reset(new VarianceScalingSimulator(1.f, data_simu::Mode_t::Fan_avg,
                                            data_simu::Distribution_t::Norm, bottom_dim, top_dim));
  } else if (1 == index) {
    float stddev = sqrt(1.f / top_dim);
    simu.reset(new GaussianDataSimulator(0, stddev, -2 * stddev, 2 * stddev));
  } else {
    CK_THROW_(Error_t::OutOfBound, "index != {0, 1}.");
  }

  return simu;
}

template class FullyConnectedLayer<float>;

}  // namespace HugeCTR

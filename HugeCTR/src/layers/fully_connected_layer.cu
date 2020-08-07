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

#include <linalg/matrix_vector_op.cuh>
#include <linalg/reduce.cuh>
#include <layers/fully_connected_layer.hpp>
#include <utils.cuh>
#include <math.h>
#include <vector>
#include <data_parser.hpp>

namespace HugeCTR {

FullyConnectedLayer::FullyConnectedLayer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                                         const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
                                         const std::shared_ptr<Tensor<float>>& in_tensor,
                                         const std::shared_ptr<Tensor<float>>& out_tensor,
                                         TensorFormat_t weight_format,
                                         cublasHandle_t const& cublas_handle, int device_id,
                                         bool use_mixed_precision,
                                         std::vector<Initializer_t> initializer_types)
    : cublas_handle_(cublas_handle),
      Layer(device_id, initializer_types),
      use_mixed_precision_(use_mixed_precision) {
  try {
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor->get_dims();
    const auto& out_tensor_dim = out_tensor->get_dims();
    // 1. two dim?
    if (in_tensor_dim.size() != 2 || out_tensor_dim.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
    }
    // 2. dim match?
    assert(in_tensor->get_format() == TensorFormat_t::WH ||
           in_tensor->get_format() == TensorFormat_t::HW);
    assert(out_tensor->get_format() == TensorFormat_t::WH ||
           out_tensor->get_format() == TensorFormat_t::HW);
    size_t m = in_tensor->get_format() == TensorFormat_t::WH ? in_tensor_dim[1] : in_tensor_dim[0];
    size_t n =
        out_tensor->get_format() == TensorFormat_t::WH ? out_tensor_dim[0] : out_tensor_dim[1];
    size_t k = in_tensor->get_format() == TensorFormat_t::WH ? in_tensor_dim[0] : in_tensor_dim[1];
    size_t m_ck =
        out_tensor->get_format() == TensorFormat_t::WH ? out_tensor_dim[1] : out_tensor_dim[0];
    if (m != m_ck) {
      CK_THROW_(Error_t::WrongInput, "size of input / output tensor doesn't match");
    }

    std::vector<size_t> weight_dim;
    std::vector<size_t> bias_dim;
    if (weight_format == TensorFormat_t::WH) {
      weight_dim = {n, k};
      bias_dim = {n, 1};
    } else if (weight_format == TensorFormat_t::HW) {
      weight_dim = {k, n};
      bias_dim = {1, n};
    } else {
      CK_THROW_(Error_t::WrongInput, "weight_format doesn't match Mlp Layer");
    }

    weights_.emplace_back(new Tensor<float>(weight_dim, weight_buff, weight_format));
    weights_.emplace_back(new Tensor<float>(bias_dim, weight_buff, weight_format));
    wgrad_.emplace_back(new Tensor<float>(weight_dim, wgrad_buff, weight_format));
    wgrad_.emplace_back(new Tensor<float>(bias_dim, wgrad_buff, weight_format));
    in_tensors_.emplace_back(in_tensor);
    out_tensors_.emplace_back(out_tensor);
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


void FullyConnectedLayer::fprop(cudaStream_t stream) {
  CK_CUBLAS_THROW_(cublasSetStream(cublas_handle_, stream));
  CudaDeviceContext context(get_device_id());

  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];

  float* weight = (weights_[0])->get_ptr();
  float* bias = (weights_[1])->get_ptr();
  float* in = in_tensor->get_ptr();
  float* out = out_tensor->get_ptr();

  const auto& in_tensor_dim = in_tensor->get_dims();
  const auto& out_tensor_dim = out_tensor->get_dims();

  int m, n, k;

  m = in_tensor->get_format() == TensorFormat_t::WH ? in_tensor_dim[1] : in_tensor_dim[0];
  n = out_tensor->get_format() == TensorFormat_t::WH ? out_tensor_dim[0] : out_tensor_dim[1];
  k = in_tensor->get_format() == TensorFormat_t::WH ? in_tensor_dim[0] : in_tensor_dim[1];

  float alpha = 1.0f, beta = 0.0f;

  if ((weights_[0])->get_format() == TensorFormat_t::HW &&
      in_tensor->get_format() == TensorFormat_t::HW &&
      out_tensor->get_format() == TensorFormat_t::HW) {
    CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, weight,
                                  CUDA_R_32F, n, in, CUDA_R_32F, k, &beta, out, CUDA_R_32F, n,
                                  CUDA_R_32F, falgo_));
    //MLCommon::LinAlg::matrixVectorOp(out, out, bias, n, m, true, true,
    //              [] __device__(float a, float b) { return a + b; }, stream);
    add_bias(out, bias, m, n, true, stream);
  } else if ((weights_[0])->get_format() == TensorFormat_t::WH &&
             in_tensor->get_format() == TensorFormat_t::WH &&
             out_tensor->get_format() == TensorFormat_t::WH) {
    CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, in,
                                  CUDA_R_32F, m, weight, CUDA_R_32F, k, &beta, out, CUDA_R_32F, m,
                                  CUDA_R_32F, falgo_));
    //MLCommon::LinAlg::matrixVectorOp(out, out, bias, n, m, false, true,
    //          [] __device__(float a, float b) { return a + b; }, stream);
    add_bias(out, bias, m, n, false, stream);
  } else
    CK_THROW_(Error_t::UnSupportedFormat, "The format combination is not supported");
}

void FullyConnectedLayer::bprop(cudaStream_t stream) {
  CK_CUBLAS_THROW_(cublasSetStream(cublas_handle_, stream));

  CudaDeviceContext context(get_device_id());

  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];

  float* wgrad = (wgrad_[0])->get_ptr();
  float* bias_grad = (wgrad_[1])->get_ptr();
  float* weight = (weights_[0])->get_ptr();
  float* in = in_tensor->get_ptr();
  float* out = out_tensor->get_ptr();

  const auto& in_tensor_dim = in_tensor->get_dims();
  const auto& out_tensor_dim = out_tensor->get_dims();

  int m, n, k;

  m = in_tensor->get_format() == TensorFormat_t::WH ? in_tensor_dim[1] : in_tensor_dim[0];
  n = out_tensor->get_format() == TensorFormat_t::WH ? out_tensor_dim[0] : out_tensor_dim[1];
  k = in_tensor->get_format() == TensorFormat_t::WH ? in_tensor_dim[0] : in_tensor_dim[1];

  float alpha = 1.0f, beta_w = 1.0f, beta_x = 0.0f;
  // row-major
  if ((wgrad_[0])->get_format() == TensorFormat_t::HW &&
      in_tensor->get_format() == TensorFormat_t::HW &&
      out_tensor->get_format() == TensorFormat_t::HW) {
    // gradient respect to W
    CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, out,
                                  CUDA_R_32F, n, in, CUDA_R_32F, k, &beta_w, wgrad, CUDA_R_32F, n,
                                  CUDA_R_32F, balgo_W_));
    // gradient respect to Xn
    CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, weight,
                                  CUDA_R_32F, n, out, CUDA_R_32F, n, &beta_x, in, CUDA_R_32F, k,
                                  CUDA_R_32F, balgo_Xn_));
    MLCommon::LinAlg::reduce(bias_grad, out, m, n, float(0), false, true, stream, true);
  }
  // Col-major
  else if ((weights_[0])->get_format() == TensorFormat_t::WH &&
           in_tensor->get_format() == TensorFormat_t::WH &&
           out_tensor->get_format() == TensorFormat_t::WH) {
    // gradient respect to W
    CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &alpha, in,
                                  CUDA_R_32F, m, out, CUDA_R_32F, m, &beta_w, wgrad, CUDA_R_32F, k,
                                  CUDA_R_32F, balgo_W_));
    // gradient respect to Xn
    CK_CUBLAS_THROW_(cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &alpha, out,
                                  CUDA_R_32F, m, weight, CUDA_R_32F, k, &beta_x, in, CUDA_R_32F, m,
                                  CUDA_R_32F, balgo_Xn_));
    MLCommon::LinAlg::reduce(bias_grad, out, m, n, float(0), true, true, stream, true);
  } else
    CK_THROW_(Error_t::UnSupportedFormat, "The format combination is not supported");
}

void FullyConnectedLayer::search_algorithm() {
  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());
  const int repeat_num = 5;

  // CUDA stream to be used for cublas on this device
  cudaStream_t stream;
  CK_CUDA_THROW_(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Set stream to cublas handler
  CK_CUBLAS_THROW_(cublasSetStream(cublas_handle_, stream));

  // Device Tensors to be used
  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];
  float* weight = (weights_[0])->get_ptr();
  float* in = in_tensor->get_ptr();
  float* out = out_tensor->get_ptr();
  float* wgrad = (wgrad_[0])->get_ptr();

  // Tensor dim
  const auto& in_tensor_dim = in_tensor->get_dims();
  const auto& out_tensor_dim = out_tensor->get_dims();

  int m, n, k;
  m = in_tensor->get_format() == TensorFormat_t::WH ? in_tensor_dim[1] : in_tensor_dim[0];
  n = out_tensor->get_format() == TensorFormat_t::WH ? out_tensor_dim[0] : out_tensor_dim[1];
  k = in_tensor->get_format() == TensorFormat_t::WH ? in_tensor_dim[0] : in_tensor_dim[1];

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

  // Search all the algorithm for fprop
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta = 0.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, stream));
    for (int i = 0; i < repeat_num; ++i) {
      if ((weights_[0])->get_format() == TensorFormat_t::HW &&
          in_tensor->get_format() == TensorFormat_t::HW &&
          out_tensor->get_format() == TensorFormat_t::HW) {
        status = cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, weight,
                              CUDA_R_32F, n, in, CUDA_R_32F, k, &beta, out, CUDA_R_32F, n,
                              CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
      } else if ((weights_[0])->get_format() == TensorFormat_t::WH &&
                 in_tensor->get_format() == TensorFormat_t::WH &&
                 out_tensor->get_format() == TensorFormat_t::WH) {
        status = cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, in,
                              CUDA_R_32F, m, weight, CUDA_R_32F, k, &beta, out, CUDA_R_32F, m,
                              CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
      } else
        CK_THROW_(Error_t::UnSupportedFormat, "The format combination is not supported");
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
      falgo_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = 100000000.0;

  // Search all the algorithm for bprop_W
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta_w = 1.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, stream));
    for (int i = 0; i < repeat_num; ++i) {
      if ((wgrad_[0])->get_format() == TensorFormat_t::HW &&
          in_tensor->get_format() == TensorFormat_t::HW &&
          out_tensor->get_format() == TensorFormat_t::HW) {
        status = cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, out,
                              CUDA_R_32F, n, in, CUDA_R_32F, k, &beta_w, wgrad, CUDA_R_32F, n,
                              CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
      } else if ((weights_[0])->get_format() == TensorFormat_t::WH &&
                 in_tensor->get_format() == TensorFormat_t::WH &&
                 out_tensor->get_format() == TensorFormat_t::WH) {
        status = cublasGemmEx(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &alpha, in,
                              CUDA_R_32F, m, out, CUDA_R_32F, m, &beta_w, wgrad, CUDA_R_32F, k,
                              CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
      } else
        CK_THROW_(Error_t::UnSupportedFormat, "The format combination is not supported");
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
      balgo_W_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = 100000000.0;

  // Search all the algorithm for bprop_Xn
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta_x = 0.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, stream));
    for (int i = 0; i < repeat_num; ++i) {
      if ((wgrad_[0])->get_format() == TensorFormat_t::HW &&
          in_tensor->get_format() == TensorFormat_t::HW &&
          out_tensor->get_format() == TensorFormat_t::HW) {
        status = cublasGemmEx(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, weight,
                              CUDA_R_32F, n, out, CUDA_R_32F, n, &beta_x, in, CUDA_R_32F, k,
                              CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
      } else if ((weights_[0])->get_format() == TensorFormat_t::WH &&
                 in_tensor->get_format() == TensorFormat_t::WH &&
                 out_tensor->get_format() == TensorFormat_t::WH) {
        status = cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &alpha, out,
                              CUDA_R_32F, m, weight, CUDA_R_32F, k, &beta_x, in, CUDA_R_32F, m,
                              CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
      } else
        CK_THROW_(Error_t::UnSupportedFormat, "The format combination is not supported");
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
  CK_CUDA_THROW_(cudaStreamDestroy(stream));
}

std::unique_ptr<DataSimulator<float>> FullyConnectedLayer::get_uniform_initializer(
    const int index) {
  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor->get_format() == TensorFormat_t::WH ? (in_tensor->get_dims())[0]
                                                                   : (in_tensor->get_dims())[1];
  float top_dim = out_tensor->get_format() == TensorFormat_t::WH ? (out_tensor->get_dims())[0]
                                                                 : (out_tensor->get_dims())[1];

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::unique_ptr<DataSimulator<float>>(new UnifiedDataSimulator<float>(-1 * limit, limit));
}

std::unique_ptr<DataSimulator<float>> FullyConnectedLayer::get_xavier_uniform_initializer(
    const int index) {
  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor->get_format() == TensorFormat_t::WH ? (in_tensor->get_dims())[0]
                                                                   : (in_tensor->get_dims())[1];
  float top_dim = out_tensor->get_format() == TensorFormat_t::WH ? (out_tensor->get_dims())[0]
                                                                 : (out_tensor->get_dims())[1];

  return std::unique_ptr<DataSimulator<float>>(new VarianceScalingSimulator<float>(
      1.f, data_simu::Mode_t::Fan_avg, data_simu::Distribution_t::Uniform,
      0 == index ? bottom_dim : 0, top_dim));
}

std::unique_ptr<DataSimulator<float>> FullyConnectedLayer::get_xavier_norm_initializer(
    const int index) {
  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor->get_format() == TensorFormat_t::WH ? (in_tensor->get_dims())[0]
                                                                   : (in_tensor->get_dims())[1];
  float top_dim = out_tensor->get_format() == TensorFormat_t::WH ? (out_tensor->get_dims())[0]
                                                                 : (out_tensor->get_dims())[1];

  return std::unique_ptr<DataSimulator<float>>(new VarianceScalingSimulator<float>(
      1.f, data_simu::Mode_t::Fan_avg, data_simu::Distribution_t::Norm, 0 == index ? bottom_dim : 0,
      top_dim));
}

std::unique_ptr<DataSimulator<float>> FullyConnectedLayer::get_default_initializer(
    const int index) {
  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor->get_format() == TensorFormat_t::WH ? (in_tensor->get_dims())[0]
                                                                   : (in_tensor->get_dims())[1];
  float top_dim = out_tensor->get_format() == TensorFormat_t::WH ? (out_tensor->get_dims())[0]
                                                                 : (out_tensor->get_dims())[1];

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

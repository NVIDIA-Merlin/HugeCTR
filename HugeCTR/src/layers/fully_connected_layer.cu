/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <layers/fully_connected_layer.hpp>
#include <linalg/matrix_vector_op.cuh>
#include <linalg/reduce.cuh>
#include <utils.cuh>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

namespace {

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
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

}  // namespace

FullyConnectedLayer<float>::FullyConnectedLayer(
    const std::shared_ptr<BufferBlock2<float>>& weight_buff,
    const std::shared_ptr<BufferBlock2<float>>& wgrad_buff, const Tensor2<float>& in_tensor,
    const Tensor2<float>& out_tensor, const std::shared_ptr<GPUResource>& gpu_resource,
    bool use_mixed_precision, bool enable_tf32_compute,
    std::vector<Initializer_t> initializer_types)
    : TrainableLayer<float>(weight_buff, weight_buff, wgrad_buff, gpu_resource, initializer_types),
      use_mixed_precision_(use_mixed_precision),
      enable_tf32_compute_(enable_tf32_compute) {
  try {
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor.get_dimensions();
    const auto& out_tensor_dim = out_tensor.get_dimensions();
    // 1. input and output have the same dim
    if (in_tensor_dim.size() != out_tensor_dim.size()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "input and output tensor don't have same dimensions");
    }
    // 2. dim match?
    size_t in_batch_size = 1;
    size_t out_batch_size = 1;
    size_t input_size = in_tensor_dim[in_tensor_dim.size() - 1];
    size_t output_size = out_tensor_dim[out_tensor_dim.size() - 1];

    for (size_t idx = 0; idx < in_tensor_dim.size() - 1; idx++) {
      in_batch_size = in_batch_size * in_tensor_dim[idx];
      out_batch_size = out_batch_size * out_tensor_dim[idx];
    }

    if (in_batch_size != out_batch_size) {
      HCTR_OWN_THROW(Error_t::WrongInput, "size of input / output tensor doesn't match");
    }

    std::vector<size_t> weight_dim = {input_size, output_size};
    std::vector<size_t> bias_dim = {1, output_size};

    this->set_weight(0, weight_dim);
    this->set_weight(1, bias_dim);
    this->set_wgrad(0, weight_dim);
    this->set_wgrad(1, bias_dim);

    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);
    // Where should we create this cuBLAS handle?
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

void FullyConnectedLayer<float>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  Tensor2<float>& in_tensor = get_in_tensors(is_train)[0];
  Tensor2<float>& out_tensor = out_tensors_[0];

  float* weight = this->get_weight(0).get_ptr();
  float* bias = this->get_weight(1).get_ptr();
  float* in = in_tensor.get_ptr();
  float* out = out_tensor.get_ptr();

  const auto& in_tensor_dim = in_tensor.get_dimensions();
  const auto& out_tensor_dim = out_tensor.get_dimensions();

  size_t in_batch_size = 1;
  size_t input_size = in_tensor_dim[in_tensor_dim.size() - 1];
  size_t output_size = out_tensor_dim[out_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < in_tensor_dim.size() - 1; idx++) {
    in_batch_size = in_batch_size * in_tensor_dim[idx];
  }

  float alpha = 1.0f, beta = 0.0f;

  const cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, output_size,
                              in_batch_size, input_size, &alpha, weight, CUDA_R_32F, output_size,
                              in, CUDA_R_32F, input_size, &beta, out, CUDA_R_32F, output_size,
                              compute_type, falgo_));
  add_bias(out, bias, in_batch_size, output_size, true, get_gpu().get_stream());
}

void FullyConnectedLayer<float>::bprop() {
  CudaDeviceContext context(get_device_id());

  Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  Tensor2<float>& out_tensor = out_tensors_[0];

  float* wgrad = this->get_wgrad(0).get_ptr();
  float* bias_grad = this->get_wgrad(1).get_ptr();
  float* weight = this->get_weight(0).get_ptr();
  float* in = in_tensor.get_ptr();
  float* out = out_tensor.get_ptr();

  const auto& in_tensor_dim = in_tensor.get_dimensions();
  const auto& out_tensor_dim = out_tensor.get_dimensions();

  size_t in_batch_size = 1;
  size_t input_size = in_tensor_dim[in_tensor_dim.size() - 1];
  size_t output_size = out_tensor_dim[out_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < in_tensor_dim.size() - 1; idx++) {
    in_batch_size = in_batch_size * in_tensor_dim[idx];
  }

  float alpha = 1.0f, beta_w = 1.0f, beta_x = 0.0f;

  const cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  // gradient respect to W
  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, output_size,
                              input_size, in_batch_size, &alpha, out, CUDA_R_32F, output_size, in,
                              CUDA_R_32F, input_size, &beta_w, wgrad, CUDA_R_32F, output_size,
                              compute_type, balgo_W_));
  // gradient respect to Xn
  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, input_size,
                              in_batch_size, output_size, &alpha, weight, CUDA_R_32F, output_size,
                              out, CUDA_R_32F, output_size, &beta_x, in, CUDA_R_32F, input_size,
                              compute_type, balgo_Xn_));
  MLCommon::LinAlg::reduce(bias_grad, out, in_batch_size, output_size, float(0), false, true,
                           get_gpu().get_stream(), true);
}

void FullyConnectedLayer<float>::search_algorithm() {
  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());

  const int repeat_num = 100;

  // Device Tensors to be used
  Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  Tensor2<float>& out_tensor = out_tensors_[0];
  float* weight = this->get_weight(0).get_ptr();
  float* in = in_tensor.get_ptr();
  float* out = out_tensor.get_ptr();
  float* wgrad = this->get_wgrad(0).get_ptr();

  // Tensor dim
  const auto& in_tensor_dim = in_tensor.get_dimensions();
  const auto& out_tensor_dim = out_tensor.get_dimensions();

  size_t in_batch_size = 1;
  size_t out_batch_size = 1;
  size_t input_size = in_tensor_dim[in_tensor_dim.size() - 1];
  size_t output_size = out_tensor_dim[out_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < in_tensor_dim.size() - 1; idx++) {
    in_batch_size = in_batch_size * in_tensor_dim[idx];
    out_batch_size = out_batch_size * out_tensor_dim[idx];
  }

  // Record time for each algorithm
  float shortestTime = 100000000.0;
  float time;
  cudaEvent_t start, stop;
  HCTR_LIB_THROW(cudaEventCreate(&start));
  HCTR_LIB_THROW(cudaEventCreate(&stop));

  // cublas ret status
  cublasStatus_t status;

  // Start, end for search
  int startAlgo, endAlgo;
  if (use_mixed_precision_) {
    startAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    endAlgo = CUBLAS_GEMM_ALGO15_TENSOR_OP;
  } else {
    startAlgo = CUBLAS_GEMM_DEFAULT;
    endAlgo = CUBLAS_GEMM_ALGO23;
  }

  const cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  // Search all the algorithm for fprop
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta = 0.0f;

    // Record start event
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (int i = 0; i < repeat_num; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, output_size,
                            in_batch_size, input_size, &alpha, weight, CUDA_R_32F, output_size, in,
                            CUDA_R_32F, input_size, &beta, out, CUDA_R_32F, output_size,
                            compute_type, static_cast<cublasGemmAlgo_t>(testAlgo));
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
      falgo_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = 100000000.0;

  // Search all the algorithm for bprop_W
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta_w = 1.0f;

    // Record start event
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (int i = 0; i < repeat_num; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, output_size,
                            input_size, in_batch_size, &alpha, out, CUDA_R_32F, output_size, in,
                            CUDA_R_32F, input_size, &beta_w, wgrad, CUDA_R_32F, output_size,
                            compute_type, static_cast<cublasGemmAlgo_t>(testAlgo));
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
      balgo_W_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = 100000000.0;

  // Search all the algorithm for bprop_Xn
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta_x = 0.0f;

    // Record start event
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (int i = 0; i < repeat_num; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, input_size,
                            in_batch_size, output_size, &alpha, weight, CUDA_R_32F, output_size,
                            out, CUDA_R_32F, output_size, &beta_x, in, CUDA_R_32F, input_size,
                            compute_type, static_cast<cublasGemmAlgo_t>(testAlgo));
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
      balgo_Xn_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Print selection information
  // HCTR_LOG(INFO, WORLD, "The algorithm selection for fprop, bprop_W and bprop_Xn are: %d, %d and
  // %d.\n",
  //       (int)falgo_, (int)balgo_W_, (int)balgo_Xn_);

  // Output msg
  // HCTR_LOG(INFO, ROOT, "The fully-connected layer has finished choosing the algorithm for cublas
  // Gemm.\n"); Clean-up
  HCTR_LIB_THROW(cudaEventDestroy(start));
  HCTR_LIB_THROW(cudaEventDestroy(stop));
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<float>::get_uniform_initializer(
    const int index) {
  const Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  const Tensor2<float>& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor.get_dimensions()[in_tensor.get_dimensions().size() - 1];
  float top_dim = out_tensor.get_dimensions()[out_tensor.get_dimensions().size() - 1];

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<float>::get_xavier_uniform_initializer(
    const int index) {
  const Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  const Tensor2<float>& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor.get_dimensions()[in_tensor.get_dimensions().size() - 1];
  float top_dim = out_tensor.get_dimensions()[out_tensor.get_dimensions().size() - 1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<float>::get_xavier_norm_initializer(
    const int index) {
  const Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  const Tensor2<float>& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor.get_dimensions()[in_tensor.get_dimensions().size() - 1];
  float top_dim = out_tensor.get_dimensions()[out_tensor.get_dimensions().size() - 1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<float>::get_default_initializer(
    const int index) {
  const Tensor2<float>& in_tensor = get_in_tensors(true)[0];
  const Tensor2<float>& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor.get_dimensions()[in_tensor.get_dimensions().size() - 1];
  float top_dim = out_tensor.get_dimensions()[out_tensor.get_dimensions().size() - 1];

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

template class FullyConnectedLayer<float>;

Core23TempFullyConnectedLayer<float>::Core23TempFullyConnectedLayer(
    const core23::Tensor& in_tensor, const core23::Tensor& out_tensor,
    const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision,
    bool enable_tf32_compute, std::vector<Initializer_t> initializer_types)
    : Core23TempTrainableLayer<float>({in_tensor}, {out_tensor}, gpu_resource, initializer_types),
      use_mixed_precision_(use_mixed_precision),
      enable_tf32_compute_(enable_tf32_compute) {
  try {
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor.shape();
    const auto& out_tensor_dim = out_tensor.shape();
    // 1. input and output have the same dim
    if (in_tensor_dim.dims() != out_tensor_dim.dims()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "input and output tensor don't have same dimensions");
    }
    // 2. dim match?
    int64_t in_batch_size = 1;
    int64_t out_batch_size = 1;
    int64_t input_size = in_tensor_dim.size(in_tensor_dim.dims() - 1);
    int64_t output_size = out_tensor_dim.size(out_tensor_dim.dims() - 1);

    for (int64_t idx = 0; idx < in_tensor_dim.dims() - 1; idx++) {
      in_batch_size = in_batch_size * in_tensor_dim.size(idx);
      out_batch_size = out_batch_size * out_tensor_dim.size(idx);
    }

    if (in_batch_size != out_batch_size) {
      HCTR_OWN_THROW(Error_t::WrongInput, "size of input / output tensor doesn't match");
    }

    core23::Shape weight_dim = {input_size, output_size};
    core23::Shape bias_dim = {1, output_size};

    this->set_weight(0, weight_dim);
    this->set_weight(1, bias_dim);
    this->set_wgrad(0, weight_dim);
    this->set_wgrad(1, bias_dim);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

void Core23TempFullyConnectedLayer<float>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  core23::Tensor& in_tensor = get_in_tensors(is_train)[0];
  core23::Tensor& out_tensor = this->output_tensors_[0];

  float* weight = this->get_weight(0).data<float>();
  float* bias = this->get_weight(1).data<float>();
  float* in = in_tensor.data<float>();
  float* out = out_tensor.data<float>();

  const auto& in_tensor_dim = in_tensor.shape();
  const auto& out_tensor_dim = out_tensor.shape();

  int64_t in_batch_size = 1;
  int64_t input_size = in_tensor_dim.size(in_tensor_dim.dims() - 1);
  int64_t output_size = out_tensor_dim.size(out_tensor_dim.dims() - 1);

  for (int64_t idx = 0; idx < in_tensor_dim.dims() - 1; idx++) {
    in_batch_size = in_batch_size * in_tensor_dim.size(idx);
  }

  float alpha = 1.0f, beta = 0.0f;

  const cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, output_size,
                              in_batch_size, input_size, &alpha, weight, CUDA_R_32F, output_size,
                              in, CUDA_R_32F, input_size, &beta, out, CUDA_R_32F, output_size,
                              compute_type, falgo_));
  add_bias(out, bias, in_batch_size, output_size, true, get_gpu().get_stream());
}

void Core23TempFullyConnectedLayer<float>::bprop() {
  CudaDeviceContext context(get_device_id());

  core23::Tensor& in_tensor = get_in_tensors(true)[0];
  core23::Tensor& out_tensor = this->output_tensors_[0];

  float* wgrad = this->get_wgrad(0).data<float>();
  float* bias_grad = this->get_wgrad(1).data<float>();
  float* weight = this->get_weight(0).data<float>();
  float* in = in_tensor.data<float>();
  float* out = out_tensor.data<float>();

  const auto& in_tensor_dim = in_tensor.shape();
  const auto& out_tensor_dim = out_tensor.shape();

  int64_t in_batch_size = 1;
  int64_t input_size = in_tensor_dim.size(in_tensor_dim.dims() - 1);
  int64_t output_size = out_tensor_dim.size(out_tensor_dim.dims() - 1);

  for (int64_t idx = 0; idx < in_tensor_dim.dims() - 1; idx++) {
    in_batch_size = in_batch_size * in_tensor_dim.size(idx);
  }

  float alpha = 1.0f, beta_w = 1.0f, beta_x = 0.0f;

  const cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  // gradient respect to W
  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, output_size,
                              input_size, in_batch_size, &alpha, out, CUDA_R_32F, output_size, in,
                              CUDA_R_32F, input_size, &beta_w, wgrad, CUDA_R_32F, output_size,
                              compute_type, balgo_W_));
  // gradient respect to Xn
  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, input_size,
                              in_batch_size, output_size, &alpha, weight, CUDA_R_32F, output_size,
                              out, CUDA_R_32F, output_size, &beta_x, in, CUDA_R_32F, input_size,
                              compute_type, balgo_Xn_));
  MLCommon::LinAlg::reduce(bias_grad, out, in_batch_size, output_size, float(0), false, true,
                           get_gpu().get_stream(), true);
}

void Core23TempFullyConnectedLayer<float>::search_algorithm() {
  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());

  const int repeat_num = 100;

  // Device Tensors to be used
  core23::Tensor& in_tensor = get_in_tensors(true)[0];
  core23::Tensor& out_tensor = output_tensors_[0];
  float* weight = this->get_weight(0).data<float>();
  float* in = in_tensor.data<float>();
  float* out = out_tensor.data<float>();
  float* wgrad = this->get_wgrad(0).data<float>();

  // Tensor dim
  const auto& in_tensor_dim = in_tensor.shape();
  const auto& out_tensor_dim = out_tensor.shape();

  int64_t in_batch_size = 1;
  int64_t out_batch_size = 1;
  int64_t input_size = in_tensor_dim.size(in_tensor_dim.dims() - 1);
  int64_t output_size = out_tensor_dim.size(out_tensor_dim.dims() - 1);

  for (int64_t idx = 0; idx < in_tensor_dim.dims() - 1; idx++) {
    in_batch_size = in_batch_size * in_tensor_dim.size(idx);
    out_batch_size = out_batch_size * out_tensor_dim.size(idx);
  }

  // Record time for each algorithm
  float shortestTime = 100000000.0;
  float time;
  cudaEvent_t start, stop;
  HCTR_LIB_THROW(cudaEventCreate(&start));
  HCTR_LIB_THROW(cudaEventCreate(&stop));

  // cublas ret status
  cublasStatus_t status;

  // Start, end for search
  int startAlgo, endAlgo;
  if (use_mixed_precision_) {
    startAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    endAlgo = CUBLAS_GEMM_ALGO15_TENSOR_OP;
  } else {
    startAlgo = CUBLAS_GEMM_DEFAULT;
    endAlgo = CUBLAS_GEMM_ALGO23;
  }

  const cublasComputeType_t compute_type =
      enable_tf32_compute_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  // Search all the algorithm for fprop
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta = 0.0f;

    // Record start event
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (int i = 0; i < repeat_num; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, output_size,
                            in_batch_size, input_size, &alpha, weight, CUDA_R_32F, output_size, in,
                            CUDA_R_32F, input_size, &beta, out, CUDA_R_32F, output_size,
                            compute_type, static_cast<cublasGemmAlgo_t>(testAlgo));
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
      falgo_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = 100000000.0;

  // Search all the algorithm for bprop_W
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta_w = 1.0f;

    // Record start event
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (int i = 0; i < repeat_num; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, output_size,
                            input_size, in_batch_size, &alpha, out, CUDA_R_32F, output_size, in,
                            CUDA_R_32F, input_size, &beta_w, wgrad, CUDA_R_32F, output_size,
                            compute_type, static_cast<cublasGemmAlgo_t>(testAlgo));
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
      balgo_W_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = 100000000.0;

  // Search all the algorithm for bprop_Xn
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    float alpha = 1.0f, beta_x = 0.0f;

    // Record start event
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (int i = 0; i < repeat_num; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, input_size,
                            in_batch_size, output_size, &alpha, weight, CUDA_R_32F, output_size,
                            out, CUDA_R_32F, output_size, &beta_x, in, CUDA_R_32F, input_size,
                            compute_type, static_cast<cublasGemmAlgo_t>(testAlgo));
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
      balgo_Xn_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Print selection information
  // HCTR_LOG(INFO, WORLD, "The algorithm selection for fprop, bprop_W and bprop_Xn are: %d, %d and
  // %d.\n",
  //       (int)falgo_, (int)balgo_W_, (int)balgo_Xn_);

  // Output msg
  // HCTR_LOG(INFO, ROOT, "The fully-connected layer has finished choosing the algorithm for cublas
  // Gemm.\n"); Clean-up
  HCTR_LIB_THROW(cudaEventDestroy(start));
  HCTR_LIB_THROW(cudaEventDestroy(stop));
}

std::unique_ptr<DataSimulator> Core23TempFullyConnectedLayer<float>::get_uniform_initializer(
    const int index) {
  const core23::Tensor& in_tensor = get_in_tensors(true)[0];
  const core23::Tensor& out_tensor = this->output_tensors_[0];
  float bottom_dim = in_tensor.shape().size(in_tensor.shape().dims() - 1);
  float top_dim = out_tensor.shape().size(out_tensor.shape().dims() - 1);

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> Core23TempFullyConnectedLayer<float>::get_xavier_uniform_initializer(
    const int index) {
  const core23::Tensor& in_tensor = get_in_tensors(true)[0];
  const core23::Tensor& out_tensor = this->output_tensors_[0];
  float bottom_dim = in_tensor.shape().size(in_tensor.shape().dims() - 1);
  float top_dim = out_tensor.shape().size(out_tensor.shape().dims() - 1);

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> Core23TempFullyConnectedLayer<float>::get_xavier_norm_initializer(
    const int index) {
  const core23::Tensor& in_tensor = get_in_tensors(true)[0];
  const core23::Tensor& out_tensor = this->output_tensors_[0];
  float bottom_dim = in_tensor.shape().size(in_tensor.shape().dims() - 1);
  float top_dim = out_tensor.shape().size(out_tensor.shape().dims() - 1);

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> Core23TempFullyConnectedLayer<float>::get_default_initializer(
    const int index) {
  const core23::Tensor& in_tensor = get_in_tensors(true)[0];
  const core23::Tensor& out_tensor = this->output_tensors_[0];
  float bottom_dim = in_tensor.shape().size(in_tensor.shape().dims() - 1);
  float top_dim = out_tensor.shape().size(out_tensor.shape().dims() - 1);

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

template class Core23TempFullyConnectedLayer<float>;

}  // namespace HugeCTR

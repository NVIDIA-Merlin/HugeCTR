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

#include <layers/fused_fully_connected_layer.hpp>
#include <network_buffer_channels.hpp>
#include <utils.cuh>
#include <utils.hpp>

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
    const std::shared_ptr<BufferBlock2<float>>& master_weights_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weights_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weights_grad_buff,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const Tensor2<__half>& bottom_tensor, const Tensor2<__half>& top_tensor,
    const std::shared_ptr<GPUResource>& gpu_resource, std::vector<Initializer_t> initializer_types)
    : TrainableLayer<__half>(master_weights_buff, weights_buff, weights_grad_buff, gpu_resource,
                             initializer_types),
      falgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_x_(CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
  const auto& bottom_tensor_dim = bottom_tensor.get_dimensions();
  const auto& top_tensor_dim = top_tensor.get_dimensions();

  if (bottom_tensor_dim.size() != 2 || top_tensor_dim.size() != 2) {
    HCTR_OWN_THROW(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
  }

  size_t batch_size = bottom_tensor_dim[0];
  size_t output_size = top_tensor_dim[1];
  size_t input_size = bottom_tensor_dim[1];

  if (batch_size % 32 != 0 || output_size % 64 != 0) {
    HCTR_OWN_THROW(
        Error_t::WrongInput,
        "The first dimension of bottom tensor must be a multiple of 32, the second dimension "
        "of top tensor must be a multiple of 64.");
  }

  std::vector<size_t> kernel_dim = {input_size, output_size};
  std::vector<size_t> bias_dim = {1, output_size};

  this->set_weight(0, kernel_dim);
  this->set_weight(1, bias_dim);
  this->set_wgrad(0, kernel_dim);
  this->set_wgrad(1, bias_dim);

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
  blobs_buff->reserve(top_tensor_.get_dimensions(), &middle_tensor_);
  blobs_buff->reserve(bias_dim, &bias_grad_tensor_);
}

void FusedFullyConnectedLayer::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = this->get_weight(0).get_ptr();
  const __half* bias = this->get_weight(1).get_ptr();
  const __half* bottom = get_bottom_tensor(is_train).get_ptr();
  __half* middle = middle_tensor_.get_ptr();
  __half* top = top_tensor_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor(is_train).get_dimensions();
  const auto& top_tensor_dim = top_tensor_.get_dimensions();

  size_t batch_size = bottom_tensor_dim[0];
  size_t output_size = top_tensor_dim[1];
  size_t input_size = bottom_tensor_dim[1];

  const float alpha = 1.0f;
  const float beta = 0.0f;

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, output_size,
                              batch_size, input_size, &alpha, kernel, CUDA_R_16F, output_size,
                              bottom, CUDA_R_16F, input_size, &beta, middle, CUDA_R_16F,
                              output_size, CUDA_R_32F, falgo_k_));

  const size_t max_threads = 1024;
  const size_t blocks = batch_size;
  const size_t threads = min(output_size / 2, max_threads);

  add_bias_and_re_kernel<<<blocks, threads, 0, get_gpu().get_stream()>>>(
      top, middle, bias, output_size / 2, output_size / 2);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

void FusedFullyConnectedLayer::bprop() {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = this->get_weight(0).get_ptr();
  const __half* top = top_tensor_.get_ptr();
  __half* kernel_grad = this->get_wgrad(0).get_ptr();
  __half* bias_grad = this->get_wgrad(1).get_ptr();
  __half* bottom = get_bottom_tensor(true).get_ptr();
  __half* middle = middle_tensor_.get_ptr();
  float* bias_grad_float = bias_grad_tensor_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor(true).get_dimensions();
  const auto& top_tensor_dim = top_tensor_.get_dimensions();

  int batch_size = bottom_tensor_dim[0];
  int output_size = top_tensor_dim[1];
  int input_size = bottom_tensor_dim[1];

  const float alpha = 1.0f;
  const float beta_k = 1.0f;
  const float beta_x = 0.0f;

  initialize_array<<<(output_size - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(
      bias_grad_float, output_size, 0.0f);

  dim3 blocks(output_size / 64, batch_size / 32);
  reverse_add_bias_and_re_kernel<32>
      <<<blocks, 512, 0, get_gpu().get_stream()>>>(bias_grad_float, middle, top, output_size / 2);

  convert_array<<<(output_size - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(
      bias_grad, bias_grad_float, output_size);

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, output_size,
                              input_size, batch_size, &alpha, middle, CUDA_R_16F, output_size,
                              bottom, CUDA_R_16F, input_size, &beta_k, kernel_grad, CUDA_R_16F,
                              output_size, CUDA_R_32F, balgo_k_));

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, input_size,
                              batch_size, output_size, &alpha, kernel, CUDA_R_16F, output_size,
                              middle, CUDA_R_16F, output_size, &beta_x, bottom, CUDA_R_16F,
                              input_size, CUDA_R_32F, balgo_x_));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

void FusedFullyConnectedLayer::search_algorithm() {
  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());

  const size_t repeat_num = 100;

  // Device Tensors to be used
  __half* bottom = get_bottom_tensor(true).get_ptr();
  __half* top = top_tensor_.get_ptr();
  __half* kernel = this->get_weight(0).get_ptr();
  __half* bias = this->get_weight(1).get_ptr();
  __half* kernel_grad = this->get_wgrad(0).get_ptr();
  __half* bias_grad = this->get_wgrad(1).get_ptr();

  // Tensor dim
  const auto& bottom_tensor_dim = get_bottom_tensor(true).get_dimensions();
  const auto& top_tensor_dim = top_tensor_.get_dimensions();

  size_t batch_size = bottom_tensor_dim[0];
  size_t output_size = top_tensor_dim[1];
  size_t input_size = bottom_tensor_dim[1];

  // Record time for each algorithm
  float shortestTime = std::numeric_limits<float>::max();
  float time;
  cudaEvent_t start, stop;
  HCTR_LIB_THROW(cudaEventCreate(&start));
  HCTR_LIB_THROW(cudaEventCreate(&stop));

  // Start, end for search
  const cublasGemmAlgo_t startAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  const cublasGemmAlgo_t endAlgo = CUBLAS_GEMM_ALGO15_TENSOR_OP;

  // Search all the algorithm for falgo_k_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    // Record start event
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, output_size,
                            batch_size, input_size, &alpha, kernel, CUDA_R_16F, output_size, bottom,
                            CUDA_R_16F, input_size, &beta, top, CUDA_R_16F, output_size, CUDA_R_32F,
                            static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for fprop, skipped.\n",
      //      testAlgo);
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
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, output_size,
                            input_size, batch_size, &alpha, top, CUDA_R_16F, output_size, bottom,
                            CUDA_R_16F, input_size, &beta, kernel_grad, CUDA_R_16F, output_size,
                            CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for bprop_W, skipped.\n",
      //      testAlgo);
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
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, input_size,
                            batch_size, output_size, &alpha, kernel, CUDA_R_16F, output_size, top,
                            CUDA_R_16F, output_size, &beta, bottom, CUDA_R_16F, input_size,
                            CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }

    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for bprop_Xn, skipped.\n",
      //      testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_x_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Print selection information
  // HCTR_LOG(INFO, WORLD, "The algorithm selection for falgo_k_, balgo_k_, balgo_x_ are: %d, %d and
  // %d.\n",
  //        (int)falgo_k_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP,
  //        (int)balgo_k_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP,
  //        (int)balgo_x_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // Output msg
  // HCTR_LOG(INFO, ROOT, "The fully-connected layer has finished choosing the algorithm for cublas
  // Gemm.\n"); Clean-up
  HCTR_LIB_THROW(cudaEventDestroy(start));
  HCTR_LIB_THROW(cudaEventDestroy(stop));
}  // namespace HugeCTR

std::unique_ptr<DataSimulator> FusedFullyConnectedLayer::get_uniform_initializer(const int index) {
  size_t bottom_dim = get_bottom_tensor(true).get_dimensions()[1];
  size_t top_dim = top_tensor_.get_dimensions()[1];

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> FusedFullyConnectedLayer::get_xavier_uniform_initializer(
    const int index) {
  size_t bottom_dim = get_bottom_tensor(true).get_dimensions()[1];
  size_t top_dim = top_tensor_.get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FusedFullyConnectedLayer::get_xavier_norm_initializer(
    const int index) {
  size_t bottom_dim = get_bottom_tensor(true).get_dimensions()[1];
  size_t top_dim = top_tensor_.get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FusedFullyConnectedLayer::get_default_initializer(const int index) {
  size_t bottom_dim = get_bottom_tensor(true).get_dimensions()[1];
  size_t top_dim = top_tensor_.get_dimensions()[1];

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

Core23TempFusedFullyConnectedLayer::Core23TempFusedFullyConnectedLayer(
    const core23::Tensor& bottom_tensor, const core23::Tensor& top_tensor,
    const std::shared_ptr<GPUResource>& gpu_resource, std::vector<Initializer_t> initializer_types)
    : Core23TempTrainableLayer<__half>(gpu_resource, initializer_types),
      falgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_x_(CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
  const auto& bottom_tensor_dim = bottom_tensor.shape();
  const auto& top_tensor_dim = top_tensor.shape();

  if (bottom_tensor_dim.dims() != 2 || top_tensor_dim.dims() != 2) {
    HCTR_OWN_THROW(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
  }

  int64_t batch_size = bottom_tensor_dim.size(0);
  int64_t output_size = top_tensor_dim.size(1);
  int64_t input_size = bottom_tensor_dim.size(1);

  if (batch_size % 32 != 0 || output_size % 64 != 0) {
    HCTR_OWN_THROW(
        Error_t::WrongInput,
        "The first dimension of bottom tensor must be a multiple of 32, the second dimension "
        "of top tensor must be a multiple of 64.");
  }

  core23::Shape kernel_dim = {input_size, output_size};
  core23::Shape bias_dim = {1, output_size};

  this->set_weight(0, kernel_dim);
  this->set_weight(1, bias_dim);
  this->set_wgrad(0, kernel_dim);
  this->set_wgrad(1, bias_dim);

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;

  core23::BufferParams blobs_buffer_params = {};
  blobs_buffer_params.channel = GetBlobsBufferChannel();
  core23::Device device(core23::DeviceType::GPU, gpu_resource->get_device_id());

  middle_tensor_ = core23::Tensor(core23::TensorParams()
                                      .data_type(core23::ToScalarType<__half>::value)
                                      .shape(top_tensor_.shape())
                                      .device(device)
                                      .buffer_params(blobs_buffer_params));

  bias_grad_tensor_ = core23::Tensor(core23::TensorParams()
                                         .data_type(core23::ToScalarType<float>::value)
                                         .shape(bias_dim)
                                         .device(device)
                                         .buffer_params(blobs_buffer_params));
}

void Core23TempFusedFullyConnectedLayer::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = this->get_weight(0).data<__half>();
  const __half* bias = this->get_weight(1).data<__half>();
  const __half* bottom = get_bottom_tensor(is_train).data<__half>();
  __half* middle = middle_tensor_.data<__half>();
  __half* top = top_tensor_.data<__half>();

  const auto& bottom_tensor_dim = get_bottom_tensor(is_train).shape();
  const auto& top_tensor_dim = top_tensor_.shape();

  int64_t batch_size = bottom_tensor_dim.size(0);
  int64_t output_size = top_tensor_dim.size(1);
  int64_t input_size = bottom_tensor_dim.size(1);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, output_size,
                              batch_size, input_size, &alpha, kernel, CUDA_R_16F, output_size,
                              bottom, CUDA_R_16F, input_size, &beta, middle, CUDA_R_16F,
                              output_size, CUDA_R_32F, falgo_k_));

  const int64_t max_threads = 1024;
  const int64_t blocks = batch_size;
  const int64_t threads = min(output_size / 2, max_threads);

  add_bias_and_re_kernel<<<blocks, threads, 0, get_gpu().get_stream()>>>(
      top, middle, bias, output_size / 2, output_size / 2);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

void Core23TempFusedFullyConnectedLayer::bprop() {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = this->get_weight(0).data<__half>();
  const __half* top = top_tensor_.data<__half>();
  __half* kernel_grad = this->get_wgrad(0).data<__half>();
  __half* bias_grad = this->get_wgrad(1).data<__half>();
  __half* bottom = get_bottom_tensor(true).data<__half>();
  __half* middle = middle_tensor_.data<__half>();
  float* bias_grad_float = bias_grad_tensor_.data<float>();

  const auto& bottom_tensor_dim = get_bottom_tensor(true).shape();
  const auto& top_tensor_dim = top_tensor_.shape();

  int batch_size = bottom_tensor_dim.size(0);
  int output_size = top_tensor_dim.size(1);
  int input_size = bottom_tensor_dim.size(1);

  const float alpha = 1.0f;
  const float beta_k = 1.0f;
  const float beta_x = 0.0f;

  initialize_array<<<(output_size - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(
      bias_grad_float, output_size, 0.0f);

  dim3 blocks(output_size / 64, batch_size / 32);
  reverse_add_bias_and_re_kernel<32>
      <<<blocks, 512, 0, get_gpu().get_stream()>>>(bias_grad_float, middle, top, output_size / 2);

  convert_array<<<(output_size - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(
      bias_grad, bias_grad_float, output_size);

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, output_size,
                              input_size, batch_size, &alpha, middle, CUDA_R_16F, output_size,
                              bottom, CUDA_R_16F, input_size, &beta_k, kernel_grad, CUDA_R_16F,
                              output_size, CUDA_R_32F, balgo_k_));

  HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, input_size,
                              batch_size, output_size, &alpha, kernel, CUDA_R_16F, output_size,
                              middle, CUDA_R_16F, output_size, &beta_x, bottom, CUDA_R_16F,
                              input_size, CUDA_R_32F, balgo_x_));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

void Core23TempFusedFullyConnectedLayer::search_algorithm() {
  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());

  const int64_t repeat_num = 100;

  // Device Tensors to be used
  __half* bottom = get_bottom_tensor(true).data<__half>();
  __half* top = top_tensor_.data<__half>();
  __half* kernel = this->get_weight(0).data<__half>();
  __half* bias = this->get_weight(1).data<__half>();
  __half* kernel_grad = this->get_wgrad(0).data<__half>();
  __half* bias_grad = this->get_wgrad(1).data<__half>();

  // Tensor dim
  const auto& bottom_tensor_dim = get_bottom_tensor(true).shape();
  const auto& top_tensor_dim = top_tensor_.shape();

  int64_t batch_size = bottom_tensor_dim.size(0);
  int64_t output_size = top_tensor_dim.size(1);
  int64_t input_size = bottom_tensor_dim.size(1);

  // Record time for each algorithm
  float shortestTime = std::numeric_limits<float>::max();
  float time;
  cudaEvent_t start, stop;
  HCTR_LIB_THROW(cudaEventCreate(&start));
  HCTR_LIB_THROW(cudaEventCreate(&stop));

  // Start, end for search
  const cublasGemmAlgo_t startAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  const cublasGemmAlgo_t endAlgo = CUBLAS_GEMM_ALGO15_TENSOR_OP;

  // Search all the algorithm for falgo_k_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    // Record start event
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (int64_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, output_size,
                            batch_size, input_size, &alpha, kernel, CUDA_R_16F, output_size, bottom,
                            CUDA_R_16F, input_size, &beta, top, CUDA_R_16F, output_size, CUDA_R_32F,
                            static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for fprop, skipped.\n",
      //      testAlgo);
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
    HCTR_LIB_THROW(cudaEventRecord(start, get_gpu().get_stream()));
    for (int64_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, output_size,
                            input_size, batch_size, &alpha, top, CUDA_R_16F, output_size, bottom,
                            CUDA_R_16F, input_size, &beta, kernel_grad, CUDA_R_16F, output_size,
                            CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for bprop_W, skipped.\n",
      //      testAlgo);
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
    for (int64_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, input_size,
                            batch_size, output_size, &alpha, kernel, CUDA_R_16F, output_size, top,
                            CUDA_R_16F, output_size, &beta, bottom, CUDA_R_16F, input_size,
                            CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }

    HCTR_LIB_THROW(cudaEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(cudaEventSynchronize(stop));
    HCTR_LIB_THROW(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for bprop_Xn, skipped.\n",
      //      testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_x_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Print selection information
  // HCTR_LOG(INFO, WORLD, "The algorithm selection for falgo_k_, balgo_k_, balgo_x_ are: %d, %d and
  // %d.\n",
  //        (int)falgo_k_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP,
  //        (int)balgo_k_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP,
  //        (int)balgo_x_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // Output msg
  // HCTR_LOG(INFO, ROOT, "The fully-connected layer has finished choosing the algorithm for cublas
  // Gemm.\n"); Clean-up
  HCTR_LIB_THROW(cudaEventDestroy(start));
  HCTR_LIB_THROW(cudaEventDestroy(stop));
}  // namespace HugeCTR

std::unique_ptr<DataSimulator> Core23TempFusedFullyConnectedLayer::get_uniform_initializer(
    const int index) {
  int64_t bottom_dim = get_bottom_tensor(true).shape().size(1);
  int64_t top_dim = top_tensor_.shape().size(1);

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> Core23TempFusedFullyConnectedLayer::get_xavier_uniform_initializer(
    const int index) {
  int64_t bottom_dim = get_bottom_tensor(true).shape().size(1);
  int64_t top_dim = top_tensor_.shape().size(1);

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> Core23TempFusedFullyConnectedLayer::get_xavier_norm_initializer(
    const int index) {
  int64_t bottom_dim = get_bottom_tensor(true).shape().size(1);
  int64_t top_dim = top_tensor_.shape().size(1);

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> Core23TempFusedFullyConnectedLayer::get_default_initializer(
    const int index) {
  int64_t bottom_dim = get_bottom_tensor(true).shape().size(1);
  int64_t top_dim = top_tensor_.shape().size(1);

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

}  // namespace HugeCTR

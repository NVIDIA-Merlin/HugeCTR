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

#include <cstdio>
#include <layers/fused_relu_bias_fully_connected_layer.hpp>
#include <linalg/reduce.cuh>
#include <utils.cuh>
#include <utils.hpp>

#include "common.hpp"
namespace HugeCTR {

namespace {

template <int BLOCK_WIDTH>
__global__ void reverse_add_bias_and_re_kernel(float* bias, __half* dRelu, __half* middle,
                                               const __half* top, int ldn) {
  __shared__ __half2 elem[32][BLOCK_WIDTH + 1];
  __shared__ __half2 accu[BLOCK_WIDTH];

  const __half2 zero = TypeFunc<__half2>::zero();

  __half2* middle2 = reinterpret_cast<__half2*>(middle);
  __half2* dRelu2 = reinterpret_cast<__half2*>(dRelu);
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

    dRelu2[gi] = t;
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

FusedReluBiasFullyConnectedLayer::FusedReluBiasFullyConnectedLayer(
    const std::shared_ptr<BufferBlock2<float>>& master_weights_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weights_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weights_grad_buff,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const Tensor2<__half>& train_in_tensor, const Tensor2<__half>& mask_in_tensor,
    const Tensor2<__half>& dRelu_in_tensor, const Tensor2<__half>& db_in_tensor,
    const Tensor2<__half>& train_out_tensor, const Tensor2<__half>& mask_out_tensor,
    const Tensor2<__half>& dRelu_out_tensor, Tensor2<__half>& db_out_tensor,
    const std::shared_ptr<GPUResource>& gpu_resource, const FcPosition_t& pos,
    const Activation_t& act, const bool& skip_dgrad, std::vector<Initializer_t> initializer_types)
    : Layer(gpu_resource, initializer_types),
      balgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_x_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_b_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      pos_(pos),
      act_(act),
      skip_dgrad_(skip_dgrad) {
  const auto& bottom_tensor_dim = train_in_tensor.get_dimensions();
  const auto& top_tensor_dim = train_out_tensor.get_dimensions();

  if (bottom_tensor_dim.size() != 2 || top_tensor_dim.size() != 2) {
    CK_THROW_(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
  }

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

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
    weights_grad_buff->reserve(bias_dim, &db_out_tensor);
    weights_grad_.push_back(db_out_tensor);
  }
  blobs_buff->reserve(identity_dim, &identity_tensor_);

  train_in_tensor_ = train_in_tensor;
  if (pos_ == FcPosition_t::Head || pos_ == FcPosition_t::Isolated)
    mask_in_tensor_ = train_in_tensor;
  else {
    mask_in_tensor_ = mask_in_tensor;
    dRelu_in_tensor_ = dRelu_in_tensor;
    db_in_tensor_ = db_in_tensor;
  }
  train_out_tensor_ = train_out_tensor;
  mask_out_tensor_ = mask_out_tensor;
  dRelu_out_tensor_ = dRelu_out_tensor;
  db_out_tensor_ = db_out_tensor;
  blobs_buff->reserve(kernel_dim, &bias_grad_tensor_);

  std::vector<size_t> mask_dim = {m, n};
  blobs_buff->reserve(mask_dim, &mask_in_tensor_temp_);
}

void FusedReluBiasFullyConnectedLayer::initialize() {
  // TODO: We need different bottom desc based on is_train or not
  const auto& bottom_tensor_dim = get_bottom_tensor_fprop(true).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();
  __half* identity = identity_tensor_.get_ptr();

  int m = bottom_tensor_dim[0];
  int n = top_tensor_dim[1];
  int k = bottom_tensor_dim[1];

  initialize_array<<<(m - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(identity, m,
                                                                            __float2half(1.0f));

  CK_CUBLAS_THROW_(cublasLtMatmulDescCreate(&cublas_op_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasOperation_t trans = CUBLAS_OP_N;
  CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(cublas_op_desc_, CUBLASLT_MATMUL_DESC_TRANSA,
                                                  &trans, sizeof(trans)));
  CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(cublas_op_desc_, CUBLASLT_MATMUL_DESC_TRANSB,
                                                  &trans, sizeof(trans)));
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_RELU_AUX_BIAS;
  if (act_ == Activation_t::None) epi = CUBLASLT_EPILOGUE_BIAS;
  CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(cublas_op_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epi, sizeof(epi)));
  const __half* bias = weights_half_[1].get_ptr();
  CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(
      cublas_op_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  if (act_ != Activation_t::None) {
    __half* reluMask = mask_out_tensor_.get_ptr();
    cublasLtMatmulDescSetAttribute(cublas_op_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                   &reluMask, sizeof(reluMask));
    long reluMaskLd = n;
    cublasLtMatmulDescSetAttribute(cublas_op_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                   &reluMaskLd, sizeof(reluMaskLd));
  }

  CK_CUBLAS_THROW_(cublasLtMatrixLayoutCreate(&cublas_kernel_desc_, CUDA_R_16F, n, k, n));
  CK_CUBLAS_THROW_(cublasLtMatrixLayoutCreate(&cublas_bottom_desc_, CUDA_R_16F, k, m, k));
  CK_CUBLAS_THROW_(cublasLtMatrixLayoutCreate(&cublas_top_desc_, CUDA_R_16F, n, m, n));

  CK_CUBLAS_THROW_(cublasLtMatmulPreferenceCreate(&cublas_preference_));

  cublaslt_workspace_size_ = 1024 * 1024 * 16;  // Set it to 8MB for now
  CK_CUDA_THROW_(cudaMalloc(&cublaslt_workspace_, cublaslt_workspace_size_));
  CK_CUBLAS_THROW_(cublasLtMatmulPreferenceSetAttribute(
      cublas_preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size_,
      sizeof(cublaslt_workspace_size_)));

  uint32_t pointer_mode = CUBLASLT_POINTER_MODE_MASK_HOST;
  CK_CUBLAS_THROW_(cublasLtMatmulPreferenceSetAttribute(cublas_preference_,
                                                        CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK,
                                                        &pointer_mode, sizeof(pointer_mode)));

  // By default set algo to best estimated heurstic
  cublasLtMatmulHeuristicResult_t heuristic_result;
  int returned_res = 0;
  CK_CUBLAS_THROW_(cublasLtMatmulAlgoGetHeuristic(
      get_gpu().get_cublaslt_handle(), cublas_op_desc_, cublas_kernel_desc_, cublas_bottom_desc_,
      cublas_top_desc_, cublas_top_desc_, cublas_preference_, 1, &heuristic_result, &returned_res));

  memcpy(&falgo_k_, &heuristic_result.algo, sizeof(falgo_k_));

  if (returned_res == 0) {
    CK_CUBLAS_THROW_(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  initialize_bprop();
}

void FusedReluBiasFullyConnectedLayer::initialize_bprop() {
  // TODO: We need different bottom desc based on is_train or not
  const auto& bottom_tensor_dim = get_bottom_tensor_fprop(true).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  CK_CUBLAS_THROW_(
      cublasLtMatmulDescCreate(&cublas_op_desc_bprop_, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasOperation_t transA = CUBLAS_OP_T;
  cublasOperation_t transB = CUBLAS_OP_N;
  CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(
      cublas_op_desc_bprop_, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(
      cublas_op_desc_bprop_, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
  if (pos_ == FcPosition_t::Head || pos_ == FcPosition_t::Isolated) {
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_DEFAULT;
    CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(
        cublas_op_desc_bprop_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  } else if (pos_ == FcPosition_t::Body || pos_ == FcPosition_t::Tail) {
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_DRELU_BGRAD;
    cublasLtMatmulDescSetAttribute(cublas_op_desc_bprop_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi,
                                   sizeof(epi));
    __half* bgrad = db_in_tensor_.get_ptr();
    cublasLtMatmulDescSetAttribute(cublas_op_desc_bprop_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad,
                                   sizeof(bgrad));
    __half* reluMask = mask_in_tensor_.get_ptr();
    cublasLtMatmulDescSetAttribute(cublas_op_desc_bprop_, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                   &reluMask, sizeof(reluMask));
    long reluMaskLd = k;
    cublasLtMatmulDescSetAttribute(cublas_op_desc_bprop_, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                   &reluMaskLd, sizeof(reluMaskLd));
  }

  CK_CUBLAS_THROW_(cublasLtMatrixLayoutCreate(&cublas_dRelu_top_desc_, CUDA_R_16F, n, m, n));
  CK_CUBLAS_THROW_(cublasLtMatrixLayoutCreate(&cublas_dRelu_bottom_desc_, CUDA_R_16F, k, m, k));

  CK_CUBLAS_THROW_(cublasLtMatmulPreferenceCreate(&cublas_preference_dRelu_));

  cublaslt_workspace_size_ = 1024 * 1024 * 8;  // Set it to 8MB for now
  CK_CUDA_THROW_(cudaMalloc(&cublaslt_workspace_dRelu_, cublaslt_workspace_size_));
  CK_CUBLAS_THROW_(cublasLtMatmulPreferenceSetAttribute(
      cublas_preference_dRelu_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size_,
      sizeof(cublaslt_workspace_size_)));

  uint32_t pointer_mode = CUBLASLT_POINTER_MODE_MASK_HOST;
  CK_CUBLAS_THROW_(cublasLtMatmulPreferenceSetAttribute(cublas_preference_,
                                                        CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK,
                                                        &pointer_mode, sizeof(pointer_mode)));

  // By default set algo to best estimated heurstic
  cublasLtMatmulHeuristicResult_t heuristic_result;
  int returned_res = 0;
  CK_CUBLAS_THROW_(cublasLtMatmulAlgoGetHeuristic(
      get_gpu().get_cublaslt_handle(), cublas_op_desc_bprop_, cublas_kernel_desc_,
      cublas_dRelu_top_desc_, cublas_dRelu_bottom_desc_, cublas_dRelu_bottom_desc_,
      cublas_preference_dRelu_, 1, &heuristic_result, &returned_res));

  memcpy(&balgo_dRelu_, &heuristic_result.algo, sizeof(balgo_dRelu_));

  if (returned_res == 0) {
    CK_CUBLAS_THROW_(CUBLAS_STATUS_NOT_SUPPORTED);
  }
}
void FusedReluBiasFullyConnectedLayer::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  PROFILE_RECORD("fused_relu_bias_fully_connected.fprop.start", get_gpu().get_stream());
  const __half* kernel = weights_half_[0].get_ptr();
  const __half* bias = weights_half_[1].get_ptr();
  const __half* bottom = get_bottom_tensor_fprop(is_train).get_ptr();
  __half* top_fprop = train_out_tensor_.get_ptr();
  __half* mask_out = mask_out_tensor_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor_fprop(is_train).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  const float alpha = 1.0f;
  const float beta = 0.0f;

  PROFILE_RECORD("fused_relu_bias_fully_connected.fprop.cublasLtMatmul.start",
                 get_gpu().get_stream());
  CK_CUBLAS_THROW_(cublasLtMatmul(
      get_gpu().get_cublaslt_handle(), cublas_op_desc_, &alpha, kernel, cublas_kernel_desc_, bottom,
      cublas_bottom_desc_, &beta, top_fprop, cublas_top_desc_, top_fprop, cublas_top_desc_,
      &falgo_k_, cublaslt_workspace_, cublaslt_workspace_size_, get_gpu().get_stream()));

  PROFILE_RECORD("fused_relu_bias_fully_connected.fprop.cublasLtMatmul.stop",
                 get_gpu().get_stream());
  if ((pos_ == FcPosition_t::Tail || pos_ == FcPosition_t::Isolated) &&
      act_ != Activation_t::None) {
    size_t len = train_out_tensor_.get_num_elements();
    CK_CUDA_THROW_(cudaMemcpyAsync(mask_out, top_fprop, len * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, get_gpu().get_stream()));
  }
  PROFILE_RECORD("fused_relu_bias_fully_connected.fprop.stop", get_gpu().get_stream());
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif

}

void FusedReluBiasFullyConnectedLayer::bprop() {
  CudaDeviceContext context(get_device_id());

  PROFILE_RECORD("fused_relu_bias_fully_connected.bprop.start", get_gpu().get_stream());
  const __half* kernel = weights_half_[0].get_ptr();
  __half* mask_in = mask_in_tensor_.get_ptr();
  const __half* train_out = train_out_tensor_.get_ptr();
  __half* mask_out = mask_out_tensor_.get_ptr();
  __half* kernel_grad = weights_grad_[0].get_ptr();
  __half* bias_grad = weights_grad_[1].get_ptr();
  const __half* bottom = get_bottom_tensor_fprop(true).get_ptr();
  __half* bottom_bprop = get_bottom_tensor_bprop(true).get_ptr();
  float* bias_grad_float = bias_grad_tensor_.get_ptr();
  __half* dRelu_top = dRelu_out_tensor_.get_ptr();
  const __half* identity = identity_tensor_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor_bprop(true).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();

  int m = bottom_tensor_dim[0];
  int n = top_tensor_dim[1];
  int k = bottom_tensor_dim[1];

  const float alpha = 1.0f;
  const float beta_k = 1.0f;
  const float beta_x = 0.0f;
  const float beta_b = 0.0f;

  PROFILE_RECORD("fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel.start",
                 get_gpu().get_stream());
  if (pos_ == FcPosition_t::Tail || pos_ == FcPosition_t::Isolated) {
    if (act_ == Activation_t::None) {
      CK_CUBLAS_THROW_(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, 1,
                                    m, &alpha, train_out, CUDA_R_16F, n, identity, CUDA_R_16F, m,
                                    &beta_b, bias_grad, CUDA_R_16F, n, CUDA_R_32F, balgo_b_));

    } else {
      initialize_array<<<(n - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(bias_grad_float, n,
                                                                                0.0f);
      dim3 blocks(n / 64, m / 32);
      reverse_add_bias_and_re_kernel<32><<<blocks, 512, 0, get_gpu().get_stream()>>>(
          bias_grad_float, dRelu_top, mask_out, train_out, n / 2);
      convert_array<<<(n - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(bias_grad,
                                                                             bias_grad_float, n);
    }
  }
  PROFILE_RECORD("fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel.stop",
                 get_gpu().get_stream());

  if (act_ == Activation_t::None) {
    dRelu_top = train_out_tensor_.get_ptr();
  }

  PROFILE_RECORD("fused_relu_bias_fully_connected.bprop.cublasGemmEx_1.start",
                 get_gpu().get_stream());
  CK_CUBLAS_THROW_(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
                                &alpha, dRelu_top, CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta_k,
                                kernel_grad, CUDA_R_16F, n, CUDA_R_32F, balgo_k_));
  PROFILE_RECORD("fused_relu_bias_fully_connected.bprop.cublasGemmEx_1.stop",
                 get_gpu().get_stream());

  if (skip_dgrad_) {
    PROFILE_RECORD("fused_relu_bias_fully_connected.bprop.stop", get_gpu().get_stream());
    return;
  }

  if (pos_ == FcPosition_t::Body || pos_ == FcPosition_t::Tail) {
    bottom_bprop = dRelu_in_tensor_.get_ptr();
  }
  PROFILE_RECORD("fused_relu_bias_fully_connected.bprop.cublasGemmEx_2.start",
                 get_gpu().get_stream());
  CK_CUBLAS_THROW_(cublasLtMatmul(
      get_gpu().get_cublaslt_handle(), cublas_op_desc_bprop_, &alpha, kernel, cublas_kernel_desc_,
      dRelu_top, cublas_dRelu_top_desc_, &beta_x, bottom_bprop, cublas_dRelu_bottom_desc_,
      bottom_bprop, cublas_dRelu_bottom_desc_, &balgo_dRelu_, cublaslt_workspace_dRelu_,
      cublaslt_workspace_size_, get_gpu().get_stream()));
  PROFILE_RECORD("fused_relu_bias_fully_connected.bprop.cublasGemmEx_2.stop",
                 get_gpu().get_stream());

  PROFILE_RECORD("fused_relu_bias_fully_connected.bprop.stop", get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

void FusedReluBiasFullyConnectedLayer::search_algorithm() {
  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());
  const size_t repeat_num = 100;
  const int max_algo_count = 16;

  // Device Tensors to be used
  __half* bottom = get_bottom_tensor_fprop(true).get_ptr();
  __half* top = train_out_tensor_.get_ptr();
  __half* kernel = weights_half_[0].get_ptr();
  __half* bias = weights_half_[1].get_ptr();
  __half* kernel_grad = weights_grad_[0].get_ptr();
  __half* bias_grad = weights_grad_[1].get_ptr();
  __half* identity = identity_tensor_.get_ptr();

  // Tensor dim
  const auto& bottom_tensor_dim = get_bottom_tensor_fprop(true).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();

  int m = bottom_tensor_dim[0];
  int n = top_tensor_dim[1];
  int k = bottom_tensor_dim[1];

  // Record time for each algorithm
  float shortestTime = std::numeric_limits<float>::max();
  float time;
  cudaEvent_t start, stop;
  CK_CUDA_THROW_(cudaEventCreate(&start));
  CK_CUDA_THROW_(cudaEventCreate(&stop));

  cublasLtMatmulHeuristicResult_t heuristic_result[max_algo_count] = {0};
  int algo_count = 0;
  CK_CUBLAS_THROW_(cublasLtMatmulAlgoGetHeuristic(
      get_gpu().get_cublaslt_handle(), cublas_op_desc_, cublas_kernel_desc_, cublas_bottom_desc_,
      cublas_top_desc_, cublas_top_desc_, cublas_preference_, max_algo_count, heuristic_result,
      &algo_count));

  if (algo_count == 0) {
    CK_CUBLAS_THROW_(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  // if(get_device_id()==0) printf("M: %d, N: %d, K: %d\n", m, n, k);
  for (int algoIdx = 0; algoIdx < algo_count; algoIdx++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status =
          cublasLtMatmul(get_gpu().get_cublaslt_handle(), cublas_op_desc_, &alpha, kernel,
                         cublas_kernel_desc_, bottom, cublas_bottom_desc_, &beta, top,
                         cublas_top_desc_, top, cublas_top_desc_, &heuristic_result[algoIdx].algo,
                         cublaslt_workspace_, cublaslt_workspace_size_, get_gpu().get_stream());
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, get_gpu().get_stream()));
    CK_CUDA_THROW_(cudaEventSynchronize(stop));
    CK_CUDA_THROW_(cudaEventElapsedTime(&time, start, stop));

    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      printf("The algorithms %d is not supported for fprop, skipped.\n", testAlgo);
      continue;
    }

    // if(get_device_id()==0) printf("Algo: %d, wavesCount: %f, time: %f\n",
    //           (int)heuristic_result[algoIdx].algo,
    //           heuristic_result[algoIdx].wavesCount,
    //           time);
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      memcpy(&falgo_k_, &heuristic_result[algoIdx].algo, sizeof(falgo_k_));
      // if(get_device_id()==0) printf("Picked algorithm: %d", heuristic_result[algoIdx].algo);
    }
  }

  // dRelu in backward pass
  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();
  cublasLtMatmulHeuristicResult_t heuristic_result_dRelu[max_algo_count] = {0};
  int algo_count_dRelu = 0;
  CK_CUBLAS_THROW_(cublasLtMatmulAlgoGetHeuristic(
      get_gpu().get_cublaslt_handle(), cublas_op_desc_bprop_, cublas_kernel_desc_,
      cublas_dRelu_top_desc_, cublas_dRelu_bottom_desc_, cublas_dRelu_bottom_desc_,
      cublas_preference_dRelu_, max_algo_count, heuristic_result_dRelu, &algo_count_dRelu));

  if (algo_count_dRelu == 0) {
    CK_CUBLAS_THROW_(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  for (int algoIdx = 0; algoIdx < algo_count_dRelu; algoIdx++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasLtMatmul(get_gpu().get_cublaslt_handle(), cublas_op_desc_bprop_, &alpha,
                              kernel, cublas_kernel_desc_, top, cublas_dRelu_top_desc_, &beta,
                              bottom, cublas_dRelu_bottom_desc_, bottom, cublas_dRelu_bottom_desc_,
                              &heuristic_result_dRelu[algoIdx].algo, cublaslt_workspace_dRelu_,
                              cublaslt_workspace_size_, get_gpu().get_stream());
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, get_gpu().get_stream()));
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
      memcpy(&balgo_dRelu_, &heuristic_result_dRelu[algoIdx].algo, sizeof(balgo_dRelu_));
    }
  }

  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

  // Start, end for search
  const cublasGemmAlgo_t startAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  const cublasGemmAlgo_t endAlgo = CUBLAS_GEMM_ALGO15_TENSOR_OP;

  // Search all the algorithm for balgo_k_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
                            &alpha, top, CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta, kernel_grad,
                            CUDA_R_16F, n, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, get_gpu().get_stream()));
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

  // Search all the algorithm for balgo_b_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, 1, m,
                            &alpha, top, CUDA_R_16F, n, identity, CUDA_R_16F, m, &beta, bias_grad,
                            CUDA_R_16F, n, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
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
      balgo_b_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }
  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

  // Search all the algorithm for balgo_x_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const __half alpha = 1.0f;
    const __half beta = 0.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, k, m, n,
                            &alpha, kernel, CUDA_R_16F, n, top, CUDA_R_16F, n, &beta, bottom,
                            CUDA_R_16F, k, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }

    CK_CUDA_THROW_(cudaEventRecord(stop, get_gpu().get_stream()));
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
  // MESSAGE_("The fully-connected layer has finished choosing the algorithm for cublas Gemm.");
  // Clean-up
  CK_CUDA_THROW_(cudaEventDestroy(start));
  CK_CUDA_THROW_(cudaEventDestroy(stop));
}  // namespace HugeCTR

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_uniform_initializer(
    const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = train_out_tensor_.get_dimensions()[1];

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_xavier_uniform_initializer(
    const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = train_out_tensor_.get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_xavier_norm_initializer(
    const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = train_out_tensor_.get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_default_initializer(
    const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = train_out_tensor_.get_dimensions()[1];

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

}  // namespace HugeCTR

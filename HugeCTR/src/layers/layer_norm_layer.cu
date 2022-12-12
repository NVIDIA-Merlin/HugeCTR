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

#include <cooperative_groups.h>

#include <algorithm>
#include <functional>
#include <layers/layer_norm_layer.hpp>
#include <string>
#include <utils.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {
#define TILE_DIM 32
#define WARP_SIZE 32
#define MAX_THREADS 512
#define MAX_WARP_NUM 32

#define MAX_NUM_STRIDE 64

namespace cg = cooperative_groups;

namespace {
#define FINAL_MASK 0xffffffff

template <typename T>
using ToStringType = typename std::conditional<std::is_same<T, __half>::value, float, T>::type;

template <typename T>
__global__ void layer_norm_kernel(T* out, const T* __restrict input, T* result_var, T* result_mean,
                                  const T* __restrict gamma, const T* __restrict beta, int batch,
                                  int hidden_dim, double eps) {
  __shared__ float s_mean;
  __shared__ float s_variance;

  float mean = 0.0f;
  float t_mean = 0.0f;
  float variance = 0.0f;
  float t_variance = 0.0f;

  float local_out = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    local_out = static_cast<float>(input[blockIdx.x * hidden_dim + idx]);
    mean += local_out;
  }
  t_mean = blockDim.x <= 32 ? warpReduceSum<float>(mean) : blockReduceSum<float>(mean);
  if (threadIdx.x == 0) s_mean = t_mean / hidden_dim;
  __syncthreads();

  float tmp = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    local_out = static_cast<float>(input[blockIdx.x * hidden_dim + idx]);
    variance += (local_out - s_mean) * (local_out - s_mean);
  }
  t_variance = blockDim.x <= 32 ? warpReduceSum<float>(variance) : blockReduceSum<float>(variance);
  if (threadIdx.x == 0) s_variance = t_variance / hidden_dim + eps;  // get epsilon
  __syncthreads();

  result_mean[blockIdx.x] = static_cast<T>(s_mean);
  result_var[blockIdx.x] = static_cast<T>(s_variance);

  for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    local_out = static_cast<float>(input[blockIdx.x * hidden_dim + idx]);
    out[blockIdx.x * hidden_dim + idx] =
        static_cast<T>(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[idx])) +
                       (float)(__ldg(&beta[idx])));
  }
}

template <typename T>
__global__ void layer_norm_backward1(const T* __restrict__ out_grad, const T* __restrict__ X_data,
                                     const T* __restrict__ vars, const T* __restrict__ means,
                                     T* __restrict__ gamma_grad, T* __restrict__ betta_grad,
                                     int batch, int hidden_dim) {
  __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = threadIdx.y * hidden_dim + idx;
  int y_stride = hidden_dim * TILE_DIM;

  int pos = blockIdx.x * TILE_DIM + threadIdx.y;
  // Loop across matrix height

  float betta_tmp = 0;
  float gamma_tmp = 0;
  for (int r = threadIdx.y; r < batch; r += TILE_DIM) {
    float grad = 0.0f;
    float val = 0.0f;
    if (idx < hidden_dim) {
      grad = (float)out_grad[offset];
      val = (float)X_data[offset];
    }
    val = (val - (float)means[r]) * rsqrtf((float)vars[r]);
    betta_tmp += grad;
    gamma_tmp += (val * grad);
    offset += y_stride;
  }

  betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
  gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

  __syncthreads();

  // Sum the shared buffer.
  float s1 = betta_buffer[threadIdx.y][threadIdx.x];
  float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

  __syncthreads();

  for (int i = 1; i < TILE_DIM; i <<= 1) {
    s1 += g.shfl_down(s1, i);
    s2 += g.shfl_down(s2, i);
  }

  if (threadIdx.x == 0 && pos < hidden_dim) {
    betta_grad[pos] = static_cast<T>(s1);
    gamma_grad[pos] = static_cast<T>(s2);
  }
}
template <typename T>
__global__ void layer_norm_backward2(const T* out_grad, T* X_vals, const T* gamma, const T* vars,
                                     const T* means, T* inp_grad, int hidden_dim) {
  int iteration_stride = blockDim.x;
  int iterations = hidden_dim / iteration_stride;

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  int row = blockIdx.x;
  int id = threadIdx.x;
  int wid = id / WARP_SIZE;
  int warp_num = (iteration_stride < hidden_dim ? iteration_stride : hidden_dim) / WARP_SIZE;

  __shared__ float partialSum[MAX_WARP_NUM];

  out_grad += (row * hidden_dim);
  X_vals += (row * hidden_dim);
  inp_grad += (row * hidden_dim);

  float vals_arr[MAX_NUM_STRIDE];
  int high_index = iterations * iteration_stride + id;
#pragma unroll
  for (int i = 0; i < iterations; i++) {
    float gamma_reg = gamma[i * iteration_stride + id];
    vals_arr[i] = out_grad[i * iteration_stride + id];
    vals_arr[i] *= gamma_reg;
  }
  // to cope with the case when hidden_dim cannot be divided by iteration_stride
  if ((high_index) < hidden_dim) {
    float gamma_reg = gamma[high_index];
    vals_arr[iterations] = out_grad[high_index];
    vals_arr[iterations] *= gamma_reg;
    iterations++;
  }

  float var_reg = vars[row];
  float mean_reg = means[row];

  float sum = 0;
  float xu[MAX_NUM_STRIDE];
  for (int i = 0; i < iterations; i++) {
    xu[i] = (X_vals[i * iteration_stride + id] - mean_reg);
    sum += vals_arr[i] * xu[i];
    vals_arr[i] *= rsqrtf(var_reg);
  }

  for (int i = 1; i < WARP_SIZE; i *= 2) {
    sum += g.shfl_down(sum, i);
  }

  if (g.thread_rank() == 0) partialSum[wid] = sum;

  __syncthreads();

  if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

  __syncthreads();

  for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

  sum = g.shfl(sum, 0);
  sum /= hidden_dim;

  // 1st part: dlxhat * dxhatx
  // 2ed part: dlvar * dvarx
  for (int i = 0; i < iterations; i++) {
    vals_arr[i] += (-sum * xu[i] * rsqrtf(var_reg) / (var_reg));
  }

  sum = 0;
  for (int i = 0; i < iterations; i++) {
    sum += vals_arr[i];
  }

  for (int i = 1; i < WARP_SIZE; i *= 2) {
    sum += g.shfl_down(sum, i);
  }

  if (g.thread_rank() == 0) partialSum[wid] = sum;

  __syncthreads();

  if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

  __syncthreads();

  for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

  sum = g.shfl(sum, 0);
  sum /= hidden_dim;

  iterations = hidden_dim / iteration_stride;
  for (int i = 0; i < iterations; i++) {
    inp_grad[i * iteration_stride + id] = vals_arr[i] - sum;
  }
  if ((high_index) < hidden_dim) {
    inp_grad[high_index] = vals_arr[iterations] - sum;
  }
}

template <>
__global__ void layer_norm_backward2(const __half* out_grad, __half* X_vals, const __half* gamma,
                                     const __half* vars, const __half* means, __half* inp_grad,
                                     int hidden_dim) {
  int row_stride = hidden_dim / 2;
  int iteration_stride = blockDim.x;
  int iterations = row_stride / iteration_stride;

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  int row = blockIdx.x;
  int id = threadIdx.x;
  int wid = id / WARP_SIZE;
  int warp_num = (iteration_stride < row_stride ? iteration_stride : row_stride) / WARP_SIZE;

  __shared__ float partialSum[MAX_WARP_NUM];

  half2 vals_arr[MAX_NUM_STRIDE];
  float2 vals_arr_f[MAX_NUM_STRIDE];

  half2* inp_grad_h = reinterpret_cast<half2*>(inp_grad);
  const half2* out_grad_h = reinterpret_cast<const half2*>(out_grad);
  const half2* vals_hat_h = reinterpret_cast<const half2*>(X_vals);

  inp_grad_h += (row * row_stride);
  out_grad_h += (row * row_stride);
  vals_hat_h += (row * row_stride);

  const half2* gamma_h = reinterpret_cast<const half2*>(gamma);
  int high_index = iterations * iteration_stride + id;
#pragma unroll
  for (int i = 0; i < iterations; i++) {
    half2 gamma_reg = gamma_h[i * iteration_stride + id];
    vals_arr[i] = out_grad_h[i * iteration_stride + id];
    vals_arr[i] *= gamma_reg;  // out_grad * gamma
  }
  if ((high_index) < row_stride) {
    half2 gamma_reg = gamma_h[high_index];
    vals_arr[iterations] = out_grad_h[high_index];
    vals_arr[iterations] *= gamma_reg;  // out_grad * gamma
    iterations++;
  }
  half mean_h = means[row];
  half var_h = vars[row];
  half2 var_reg = __halves2half2(var_h, var_h);
  half2 mean_reg = __halves2half2(mean_h, mean_h);
  half2 xu[MAX_NUM_STRIDE];

  float sum = 0.f;
  for (int i = 0; i < iterations; i++) {
    xu[i] = (vals_hat_h[i * iteration_stride + id] - mean_reg);
    half2 result_h = (xu[i] * vals_arr[i]);
    float2 result_f = __half22float2(result_h);
    sum += result_f.x;
    sum += result_f.y;
    vals_arr[i] *= h2rsqrt(var_reg);
  }

  for (int i = 1; i < WARP_SIZE; i *= 2) {
    sum += g.shfl_down(sum, i);
  }

  if (g.thread_rank() == 0) partialSum[wid] = sum;

  __syncthreads();

  if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

  __syncthreads();

  for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

  sum = g.shfl(sum, 0);
  sum /= (2 * row_stride);
  half2 sum_h = __float2half2_rn(sum);

  for (int i = 0; i < iterations; i++) {
    half2 xu_grad = ((-sum_h * xu[i] * h2rsqrt(var_reg)) / (var_reg));
    vals_arr_f[i] = __half22float2(vals_arr[i]);
    float2 xu_grad_f = __half22float2(xu_grad);
    vals_arr_f[i].x += xu_grad_f.x;
    vals_arr_f[i].y += xu_grad_f.y;
  }
  sum = 0.f;
  for (int i = 0; i < iterations; i++) {
    sum += (vals_arr_f[i].x);
    sum += (vals_arr_f[i].y);
  }

  for (int i = 1; i < WARP_SIZE; i *= 2) {
    sum += g.shfl_down(sum, i);
  }

  if (g.thread_rank() == 0) partialSum[wid] = sum;

  __syncthreads();

  if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
  __syncthreads();
#endif

  for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

  sum = g.shfl(sum, 0);
  sum /= (2 * row_stride);

  iterations = row_stride / iteration_stride;
  for (int i = 0; i < iterations; i++) {
    half2 input = vals_hat_h[i * iteration_stride + id];
    vals_arr_f[i].x -= sum;
    vals_arr_f[i].y -= sum;

    half2 temp = __float22half2_rn(vals_arr_f[i]);
    inp_grad_h[i * iteration_stride + id] = temp;
  }
  if ((high_index) < row_stride) {
    vals_arr_f[iterations].x -= sum;
    vals_arr_f[iterations].y -= sum;
    half2 temp = __float22half2_rn(vals_arr_f[iterations]);
    inp_grad_h[high_index] = temp;
  }
}
}  // namespace

template <typename T>
LayerNormLayer<T>::LayerNormLayer(const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
                                  const std::shared_ptr<BufferBlock2<T>>& weight_buff,
                                  const std::shared_ptr<BufferBlock2<T>>& wgrad_buff,
                                  const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blob_buff,
                                  const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                                  const Params& params,
                                  const std::shared_ptr<GPUResource>& gpu_resource,
                                  std::vector<Initializer_t> initializer_types)
    : Base(master_weight_buff, weight_buff, wgrad_buff, gpu_resource, initializer_types),
      params_(params) {
  CudaDeviceContext context(this->get_device_id());
  const auto& in_tensor_dim = in_tensor.get_dimensions();
  const auto& out_tensor_dim = out_tensor.get_dimensions();

  assert(in_tensor_dim.size() == out_tensor_dim.size());
  if (in_tensor_dim.size() > 4 || in_tensor_dim.size() < 2) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Only 2D 3D 4D tensors can be layer-normed");
  }
  for (size_t idx = 0; idx < in_tensor_dim.size(); idx++) {
    assert(in_tensor_dim[idx] == out_tensor_dim[idx]);
  }

  size_t batch = 1;
  size_t hidden_dim = in_tensor_dim[in_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < in_tensor_dim.size() - 1; idx++) {
    batch = batch * in_tensor_dim[idx];
  }
  if (hidden_dim > static_cast<size_t>(65535)) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Unsupport hidden_dim, the last dim should not be longer than 65535");
  }

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);

  std::vector<size_t> gamma_dim = {hidden_dim, 1};
  std::vector<size_t> mean_dim = {batch, 1};

  // gamma & beta
  this->set_weight(0, gamma_dim);
  this->set_weight(1, gamma_dim);

  gamma_ = this->get_weight(0);
  beta_ = this->get_weight(1);
  // gamma grad & beta grad
  this->set_wgrad(0, gamma_dim);
  this->set_wgrad(1, gamma_dim);
  gamma_grad_ = this->get_wgrad(0);
  beta_grad_ = this->get_wgrad(1);

  // save running mean & var (cache)
  blob_buff->reserve(mean_dim, &result_save_mean_);
  blob_buff->reserve(mean_dim, &result_save_var_);
}

template <typename T>
void LayerNormLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(this->get_device_id());
  float one = 1.0f, zero = 0.0f;

  Tensor2<T>& in_tensor = in_tensors_[0];
  Tensor2<T>& out_tensor = out_tensors_[0];
  T* in = in_tensor.get_ptr();
  T* out = out_tensor.get_ptr();

  T* gamma = gamma_.get_ptr();
  T* beta = beta_.get_ptr();

  T* result_save_mean = result_save_mean_.get_ptr();
  T* result_save_var = result_save_var_.get_ptr();

  const auto& in_tensor_dim = in_tensor.get_dimensions();
  size_t batch = 1;
  size_t hidden_dim = in_tensor_dim[in_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < in_tensor_dim.size() - 1; idx++) {
    batch = batch * in_tensor_dim[idx];
  }
  dim3 block_size(min(hidden_dim, static_cast<size_t>(MAX_THREADS)), 1, 1);
  dim3 grid_size(batch, 1, 1);

  layer_norm_kernel<<<grid_size, block_size, 0, this->get_gpu().get_stream()>>>(
      out, in, result_save_var, result_save_mean, gamma, beta, batch, hidden_dim, params_.eps);
}

template <typename T>
void LayerNormLayer<T>::bprop() {
  CudaDeviceContext context(this->get_device_id());

  float one = 1.0f, zero = 0.0f;

  Tensor2<T>& in_tensor = in_tensors_[0];
  Tensor2<T>& out_tensor = out_tensors_[0];
  const auto& in_tensor_dim = in_tensor.get_dimensions();

  T* in = in_tensor.get_ptr();
  T* out = out_tensor.get_ptr();

  T* gamma = gamma_.get_ptr();

  T* gamma_grad = gamma_grad_.get_ptr();
  T* beta_grad = beta_grad_.get_ptr();

  T* result_save_mean = result_save_mean_.get_ptr();
  T* result_save_var = result_save_var_.get_ptr();

  size_t batch = 1;
  size_t hidden_dim = in_tensor_dim[in_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < in_tensor_dim.size() - 1; idx++) {
    batch = batch * in_tensor_dim[idx];
  }

  dim3 grid_dim1(max(hidden_dim / TILE_DIM, static_cast<size_t>(1)));
  dim3 block_dim1(TILE_DIM, TILE_DIM);
  layer_norm_backward1<<<grid_dim1, block_dim1, 0, this->get_gpu().get_stream()>>>(
      out, in, result_save_var, result_save_mean, gamma_grad, beta_grad, batch, hidden_dim);

  dim3 grid_dim2(batch);
  size_t blockDimx = hidden_dim < 32 ? hidden_dim : ((hidden_dim >> 5) << 5);
  dim3 block_dim2(min(blockDimx, static_cast<size_t>(MAX_THREADS)));

  layer_norm_backward2<<<grid_dim2, block_dim2, 0, this->get_gpu().get_stream()>>>(
      out, in, gamma, result_save_var, result_save_mean, in, hidden_dim);
}

template <typename T>
std::unique_ptr<DataSimulator> LayerNormLayer<T>::get_default_initializer(const int index) {
  std::unique_ptr<DataSimulator> simu;
  if (0 == index) {
    simu.reset(new ConstantDataSimulator(1.0f));
  } else if (1 == index) {
    simu.reset(new ConstantDataSimulator(0.0f));
  } else {
    HCTR_OWN_THROW(Error_t::OutOfBound, "index != {0, 1}.");
  }
  return simu;
}

template class LayerNormLayer<float>;
template class LayerNormLayer<__half>;

}  // namespace HugeCTR

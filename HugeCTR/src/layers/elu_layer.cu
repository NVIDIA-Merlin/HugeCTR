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

#include <algorithm>
#include <functional>
#include <include/utils.cuh>
#include <layers/element_wise_function.hpp>
#include <layers/elu_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/unary_op.cuh>
#include <utils.hpp>

namespace HugeCTR {

namespace {
template <typename T>
__global__ void elu_kernel(const T* input, T* output, int size, T alpha);

template <>
__global__ void elu_kernel<__half>(const __half* input, __half* output, int size, __half alpha) {
  const __half2* input2 = reinterpret_cast<const __half2*>(input);
  __half2* output2 = reinterpret_cast<__half2*>(output);
  const int size2 = size / 2;
  const __half2 alpha2 = __half2half2(alpha);

  const __half zero = __float2half(0.0f);
  const __half2 zero2 = __half2half2(zero);
  const __half one = __float2half(1.0f);
  const __half2 one2 = __half2half2(one);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size2) {
    const __half2 in2 = input2[tid];
    const __half2 pred = __hlt2(in2, zero2);
    output2[tid] = pred * (alpha2 * (h2exp(in2) - one2)) + (one2 - pred) * in2;
  }
  if (tid == 0 && size % 2 > 0) {
    const __half in = input[size - 1];
    output[size - 1] = (in < zero) ? alpha * (hexp(in) - one) : in;
  }
}

template <typename T>
__global__ void elu_dgrad_kernel(const T* d_out, T* d_in, int size, T alpha);

template <>
__global__ void elu_dgrad_kernel<__half>(const __half* d_out, __half* d_in, int size,
                                         __half alpha) {
  const __half2* d_out2 = reinterpret_cast<const __half2*>(d_out);
  __half2* d_in2 = reinterpret_cast<__half2*>(d_in);
  const int size2 = size / 2;
  const __half2 alpha2 = __half2half2(alpha);

  const __half zero = __float2half(0.0f);
  const __half2 zero2 = __half2half2(zero);
  const __half2 one2 = __float2half2_rn(1.0f);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size2) {
    const __half2 in2 = d_in2[tid];
    const __half2 out2 = d_out2[tid];
    const __half2 pred = __hlt2(in2, zero2);
    d_in2[tid] = pred * (alpha2 * h2exp(in2) * out2) + (one2 - pred) * out2;
  }
  if (tid == 0 && size % 2 > 0) {
    const __half in = d_in[size - 1];
    const __half out = d_out[size - 1];
    d_in[size - 1] = (in < zero) ? alpha * hexp(in) * out : out;
  }
}
}  // namespace

template <typename T>
EluLayer<T>::EluLayer(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor, T alpha,
                      const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource), alpha_(alpha) {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);
}

template <typename T>
void EluLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const Tensor2<T>& in_tensor = in_tensors_[0];
  Tensor2<T>& out_tensor = out_tensors_[0];

  const int len = in_tensor.get_num_elements();

  T alpha = alpha_;
  auto fop = [alpha] __device__(T in) { return (in < 0) ? alpha * (expf(in) - 1) : in; };

  MLCommon::LinAlg::unaryOp(out_tensor.get_ptr(), in_tensor.get_ptr(), len, fop,
                            get_gpu().get_stream());
}

template <>
void EluLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const Tensor2<__half>& in_tensor = in_tensors_[0];
  Tensor2<__half>& out_tensor = out_tensors_[0];

  const int len = in_tensor.get_num_elements();

  __half alpha = alpha_;
  dim3 block_size(256, 1, 1);
  dim3 grid_size((len / 2 + block_size.x - 1) / block_size.x, 1, 1);
  elu_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
      in_tensor.get_ptr(), out_tensor.get_ptr(), len, alpha);
}

template <typename T>
void EluLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  Tensor2<T>& in_tensor = in_tensors_[0];
  const Tensor2<T>& out_tensor = out_tensors_[0];

  const int len = in_tensor.get_num_elements();

  T alpha = alpha_;
  auto bop = [alpha] __device__(T d_out, T d_in) {
    return (d_in < 0) ? alpha * expf(d_in) * d_out : d_out;
  };

  MLCommon::LinAlg::binaryOp(in_tensor.get_ptr(), out_tensor.get_ptr(), in_tensor.get_ptr(), len,
                             bop, get_gpu().get_stream());
}

template <>
void EluLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());

  Tensor2<__half>& in_tensor = in_tensors_[0];
  const Tensor2<__half>& out_tensor = out_tensors_[0];

  const int len = in_tensor.get_num_elements();

  __half alpha = alpha_;
  dim3 block_size(256, 1, 1);
  dim3 grid_size((len / 2 + block_size.x - 1) / block_size.x, 1, 1);
  elu_dgrad_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
      out_tensor.get_ptr(), in_tensor.get_ptr(), len, alpha);
}

template class EluLayer<float>;
template class EluLayer<__half>;

}  // namespace HugeCTR

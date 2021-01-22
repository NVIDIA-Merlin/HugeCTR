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

#include <algorithm>
#include <functional>
#include <include/utils.cuh>
#include <layers/element_wise_function.hpp>
#include <layers/elu_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/unary_op.cuh>
#include <utils.hpp>
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

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
void EluLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const Tensor2<__half>& in_tensor = in_tensors_[0];
  Tensor2<__half>& out_tensor = out_tensors_[0];

  const int len = in_tensor.get_num_elements();

  __half alpha = alpha_;
  const __half zero = __float2half(0.f);
  const __half one = __float2half(1.f);
  auto fop = [alpha, zero, one] __device__(__half in) {
    return (in < zero) ? alpha * (hexp(in) - one) : in;
  };

  MLCommon::LinAlg::unaryOp(out_tensor.get_ptr(), in_tensor.get_ptr(), len, fop,
                            get_gpu().get_stream());
}

template <>
void EluLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());

  Tensor2<__half>& in_tensor = in_tensors_[0];
  const Tensor2<__half>& out_tensor = out_tensors_[0];

  const int len = in_tensor.get_num_elements();

  __half alpha = alpha_;
  const __half zero = __float2half(0.f);
  auto bop = [alpha, zero] __device__(__half d_out, __half d_in) {
    return (d_in < zero) ? alpha * hexp(d_in) * d_out : d_out;
  };

  MLCommon::LinAlg::binaryOp(in_tensor.get_ptr(), out_tensor.get_ptr(), in_tensor.get_ptr(), len,
                             bop, get_gpu().get_stream());
}

template class EluLayer<float>;
template class EluLayer<__half>;

}  // namespace HugeCTR

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
#include <layers/element_wise_function.hpp>
#include <layers/elu_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/unary_op.cuh>
#include <utils.hpp>
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

EluLayer::EluLayer(const Tensor2<float>& in_tensor, const Tensor2<float>& out_tensor, float alpha,
                   int device_id)
    : Layer(device_id), alpha_(alpha) {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);
}

void EluLayer::fprop(bool is_train, cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  const Tensor2<float>& in_tensor = in_tensors_[0];
  Tensor2<float>& out_tensor = out_tensors_[0];

  const int len = in_tensor.get_num_elements();

  float alpha = alpha_;
  auto fop = [alpha] __device__(float in) { return (in < 0) ? alpha * (expf(in) - 1) : in; };

  MLCommon::LinAlg::unaryOp(out_tensor.get_ptr(), in_tensor.get_ptr(), len, fop, stream);
}

void EluLayer::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  Tensor2<float>& in_tensor = in_tensors_[0];
  const Tensor2<float>& out_tensor = out_tensors_[0];

  const int len = in_tensor.get_num_elements();

  float alpha = alpha_;
  auto bop = [alpha] __device__(float d_out, float d_in) {
    return (d_in < 0) ? alpha * expf(d_in) * d_out : d_out;
  };

  MLCommon::LinAlg::binaryOp(in_tensor.get_ptr(), out_tensor.get_ptr(), in_tensor.get_ptr(), len,
                             bop, stream);
}

}  // namespace HugeCTR

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

#include <HugeCTR/include/utils.hpp>
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <functional>
#include <layers/dropout_layer.hpp>
#include <prims/linalg/binary_op.cuh>
#include <utils.cuh>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

template <typename T>
DropoutLayer<T>::DropoutLayer(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                              const std::shared_ptr<GeneralBuffer2<CudaAllocator>> blobs_buff,
                              float rate, const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource), rate_(rate), scale_(1.0 / (1.0 - rate)) {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());
  assert(rate_ > 0.f && rate_ < 1.f);

  in_tensors_.emplace_back(in_tensor);
  out_tensors_.emplace_back(out_tensor);

  blobs_buff->reserve(in_tensor.get_dimensions(), &mask_);
}

template <typename T>
void DropoutLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  if (is_train) {
    CK_CURAND_THROW_(curandGenerateUniform(get_gpu().get_replica_variant_curand_generator(), mask_.get_ptr(),
                                           in_tensors_[0].get_num_elements()));
    prop_common(in_tensors_[0].get_ptr(), out_tensors_[0].get_ptr(), get_gpu().get_stream());
  } else {
    cudaMemcpyAsync(out_tensors_[0].get_ptr(), in_tensors_[0].get_ptr(),
                    in_tensors_[0].get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                    get_gpu().get_stream());
  }
}

template <typename T>
void DropoutLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  prop_common(out_tensors_[0].get_ptr(), in_tensors_[0].get_ptr(), get_gpu().get_stream());
}

template <typename T>
void DropoutLayer<T>::prop_common(const T* in, T* out, cudaStream_t stream) {
  int len = in_tensors_[0].get_num_elements();

  float r = rate_;
  float s = scale_;
  MLCommon::LinAlg::binaryOp(out, in, mask_.get_ptr(), len,
                             [r, s] __device__(T a, float b) {
                               return TypeConvertFunc<T, float>::convert(
                                   ((1.f - b) >= r) * TypeConvertFunc<float, T>::convert(a) * s);
                             },
                             stream);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class DropoutLayer<float>;
template class DropoutLayer<__half>;

}  // namespace HugeCTR

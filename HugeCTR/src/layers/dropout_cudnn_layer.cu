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
#include <layers/dropout_cudnn_layer.hpp>
#include <prims/linalg/binary_op.cuh>
#include <utils.hpp>
#include <utils.cuh>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

template <typename T>
DropoutCudnnLayer<T>::DropoutCudnnLayer(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
               const std::shared_ptr<GeneralBuffer2<CudaAllocator>> blobs_buff, float rate,
               cudnnHandle_t const& cudnn_handle, int device_id)

    : Layer(device_id),
      rate_(rate),
      scale_(1.0 / (1.0 - rate)),
      cudnn_handle_(cudnn_handle),
      n_sms_(0) {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());
  assert(rate_ > 0.f && rate_ < 1.f);


  const auto& in_tensor_dim = in_tensor.get_dimensions();
  in_tensors_.emplace_back(in_tensor);
  out_tensors_.emplace_back(out_tensor);

  CudaDeviceContext context(get_device_id());
  
  size_t num_feature = in_tensor_dim[1];
  int batch_size = in_tensor_dim[0];
  cudnnDataType_t data_type = CudnnDataType<T>::getType();
  int n_stride = num_feature;
  int w_stride = 1;
  CK_CUDNN_THROW_(cudnnCreateTensorDescriptor(&in_out_desc_));
  CK_CUDNN_THROW_(cudnnSetTensor4dDescriptorEx(in_out_desc_, data_type, batch_size, 1, 1,
                                               num_feature, n_stride, 1, 1, w_stride));

   CK_CUDNN_THROW_(cudnnCreateDropoutDescriptor(&dropout_descriptor_));

  size_t sizeInBytes = 0;

  CK_CUDNN_THROW_(cudnnDropoutGetStatesSize(cudnn_handle_, &sizeInBytes));

  assert(sizeInBytes!=0);

  CK_CUDNN_THROW_(cudnnDropoutGetReserveSpaceSize(in_out_desc_, &reserveSpaceSizeInBytes_));

  blobs_buff->reserve({1, reserveSpaceSizeInBytes_}, &mask_);

  cudaMalloc(&cudnn_status_, sizeInBytes);
  
  CK_CUDNN_THROW_(cudnnSetDropoutDescriptor(dropout_descriptor_, cudnn_handle_, rate, cudnn_status_ ,sizeInBytes, 0));

}

template <typename T>
DropoutCudnnLayer<T>::~DropoutCudnnLayer() {
  try {
    CK_CUDNN_THROW_(cudnnDestroyDropoutDescriptor(dropout_descriptor_));
    cudaFree(cudnn_status_);
    CK_CUDNN_THROW_(cudnnDestroyTensorDescriptor(in_out_desc_));
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }

}

template <typename T>
void DropoutCudnnLayer<T>::fprop(bool is_train, cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  if (is_train) {
    CK_CUDNN_THROW_(cudnnSetStream(cudnn_handle_, stream));
    CK_CUDNN_THROW_(cudnnDropoutForward(cudnn_handle_,  dropout_descriptor_, in_out_desc_, in_tensors_[0].get_ptr(),  in_out_desc_, out_tensors_[0].get_ptr(),  mask_.get_ptr(), reserveSpaceSizeInBytes_));
  } else {
    cudaMemcpyAsync(out_tensors_[0].get_ptr(), in_tensors_[0].get_ptr(),
                    in_tensors_[0].get_size_in_bytes(), cudaMemcpyDeviceToDevice, stream);
  }
}

template <typename T>
void DropoutCudnnLayer<T>::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  CK_CUDNN_THROW_(cudnnSetStream(cudnn_handle_, stream));
  CK_CUDNN_THROW_(cudnnDropoutBackward(
    cudnn_handle_,  dropout_descriptor_, in_out_desc_, out_tensors_[0].get_ptr(),  in_out_desc_, in_tensors_[0].get_ptr(),  mask_.get_ptr(), reserveSpaceSizeInBytes_));
}

template <typename T>
int64_t DropoutCudnnLayer<T>::get_seed() const {
  FILE* f = fopen("/dev/urandom", "rb");
  if (f) {
    int64_t seed;
    size_t ret = fread(&seed, 1, sizeof(seed), f);
    fclose(f);
    if (ret == sizeof(seed)) {
      return seed;
    }
  }
  return time(nullptr);
}


template class DropoutCudnnLayer<float>;
template class DropoutCudnnLayer<__half>;

}  // namespace HugeCTR

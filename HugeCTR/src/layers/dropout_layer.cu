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

#include <HugeCTR/include/utils.hpp>
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <functional>
#include <layers/dropout_layer.hpp>
#include <network_buffer_channels.hpp>
#include <prims/linalg/binary_op.cuh>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

template <typename T>
DropoutLayer<T>::DropoutLayer(const core23::Tensor& input_tensor,
                              const core23::Tensor& output_tensor, float rate,
                              const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({input_tensor}, {output_tensor}, gpu_resource),
      rate_(rate),
      scale_(1.0 / (1.0 - rate)) {
  assert(input_tensors_[0].num_elements() == output_tensors_[0].num_elements());
  assert(rate_ > 0.f && rate_ < 1.f);

  const auto& in_tensor_shape = input_tensors_[0].shape();

  CudaDeviceContext context(get_device_id());

  size_t num_feature = in_tensor_shape.size(1);
  int batch_size = in_tensor_shape.size(0);
  cudnnDataType_t data_type = CudnnDataType<T>::getType();
  int n_stride = num_feature;
  int w_stride = 1;
  HCTR_LIB_THROW(cudnnCreateTensorDescriptor(&in_out_desc_));
  HCTR_LIB_THROW(cudnnSetTensor4dDescriptorEx(in_out_desc_, data_type, batch_size, 1, 1,
                                              num_feature, n_stride, 1, 1, w_stride));

  HCTR_LIB_THROW(cudnnCreateDropoutDescriptor(&dropout_descriptor_));

  size_t size_in_bytes = 0;

  HCTR_LIB_THROW(cudnnDropoutGetStatesSize(gpu_resource->get_cudnn_handle(), &size_in_bytes));

  assert(size_in_bytes != 0);

  HCTR_LIB_THROW(cudnnDropoutGetReserveSpaceSize(in_out_desc_, &reserveSpaceSizeInBytes_));
  core23::BufferParams buf_p{.channel = GetBlobsBufferChannel()};

  noise_mask_ = core23::Tensor(input_tensors_[0]
                                   .my_params()
                                   .shape({1, static_cast<int64_t>(reserveSpaceSizeInBytes_)})
                                   .buffer_params(buf_p));

  HCTR_LIB_THROW(cudaMalloc(&cudnn_status_, size_in_bytes));

  HCTR_LIB_THROW(cudnnSetDropoutDescriptor(dropout_descriptor_, gpu_resource->get_cudnn_handle(),
                                           rate, cudnn_status_, size_in_bytes, 0));
}

template <typename T>
DropoutLayer<T>::~DropoutLayer() {
  try {
    HCTR_LIB_THROW(cudnnDestroyDropoutDescriptor(dropout_descriptor_));
    HCTR_LIB_THROW(cudaFree(cudnn_status_));
    HCTR_LIB_THROW(cudnnDestroyTensorDescriptor(in_out_desc_));
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
}

template <typename T>
void DropoutLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  if (is_train) {
    HCTR_LIB_THROW(cudnnDropoutForward(
        get_gpu().get_cudnn_handle(), dropout_descriptor_, in_out_desc_, input_tensors_[0].data(),
        in_out_desc_, output_tensors_[0].data(), noise_mask_.data(), reserveSpaceSizeInBytes_));
  } else {
    HCTR_LIB_THROW(cudaMemcpyAsync(output_tensors_[0].data(), input_tensors_[0].data(),
                                   input_tensors_[0].num_bytes(), cudaMemcpyDeviceToDevice,
                                   get_gpu().get_stream()));
  }
}

template <typename T>
void DropoutLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  HCTR_LIB_THROW(cudnnDropoutBackward(
      get_gpu().get_cudnn_handle(), dropout_descriptor_, in_out_desc_, output_tensors_[0].data(),
      in_out_desc_, input_tensors_[0].data(), noise_mask_.data(), reserveSpaceSizeInBytes_));
}

template class DropoutLayer<float>;
template class DropoutLayer<__half>;

}  // namespace HugeCTR

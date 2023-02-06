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
#include <layers/masked_softmax_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/reduce.cuh>
#include <linalg/unary_op.cuh>
#include <utils.hpp>

namespace HugeCTR {
#define MAX_NUM_STRIDE 64

template <typename T>
MaskedSoftmaxLayer<T>::MaskedSoftmaxLayer(
    const Tensors2<T>& in_tensors, const Tensor2<T>& out_tensor, float scalar,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource), scalar_(scalar) {
  // Input 0: input data [batch_size, head, seq_len, seq_len]
  // Input 1: mask [batch_size, 1, 1, seq_len]
  assert(in_tensors[0].get_num_elements() == out_tensor.get_num_elements());
  size_t num_ = in_tensors.size();

  size_t dims_ = in_tensors[0].get_dimensions().size();
  if (num_ < 2) {
    HCTR_OWN_THROW(Error_t::WrongInput, "MaskedSoftmaxLayer needs at least 2 input tensors");
  }
  if (in_tensors[1].get_dimensions().size() != dims_) {
    HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
  }
  if (in_tensors[1].get_dimensions()[dims_ - 1] != in_tensors[0].get_dimensions()[dims_ - 1]) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "The last dimension of the input tensors should be the same");
  }

  for (size_t i = 0; i < num_; i++) {
    in_tensors_.push_back(in_tensors[i]);
  }
  out_tensors_.push_back(out_tensor);

  blobs_buff->reserve(in_tensors[0].get_dimensions(), &softmax_out_);
}

// grid = (seq_len, head_num, batch_size)
// block.x = max(32, (seq_len + 31)/32*32)
template <typename T>
void __global__ mask_softmax_fprop_kernel(T* out, T* in, const T* mask, const int batch_size,
                                          const int head_num, const int seq_len,
                                          const float scalar) {
  float data[MAX_NUM_STRIDE];
  float local_max = -1e20f;
  float local_sum = 0.0f;
  int input_offset;
  __shared__ float s_rsum, s_max;
  for (int idx = 0; blockDim.x * idx + threadIdx.x < seq_len; idx++) {
    input_offset = ((blockIdx.z * head_num + blockIdx.y) * seq_len + blockIdx.x) * seq_len +
                   blockDim.x * idx + threadIdx.x;
    int mask_offset = blockIdx.z * seq_len + blockDim.x * idx + threadIdx.x;

    float in_val = static_cast<float>(in[input_offset]);
    float mask_val = (float)mask[mask_offset];
    mask_val = (1.0f - mask_val) * 10000.0f;
    data[idx] = in_val * scalar - (float)mask_val;
    local_max = fmax(local_max, data[idx]);
  }
  float max_val = blockReduceMax<float>(local_max);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();
  for (int idx = 0; blockDim.x * idx + threadIdx.x < seq_len; idx++) {
    data[idx] = __expf(data[idx] - s_max);
    local_sum += data[idx];
  }
  float sum_val = blockReduceSum<float>(local_sum);
  if (threadIdx.x == 0) {
    s_rsum = sum_val + 1e-6f;
    s_rsum = __fdividef(1.0f, s_rsum);
  }
  __syncthreads();

  for (int idx = 0; blockDim.x * idx + threadIdx.x < seq_len; idx++) {
    input_offset = ((blockIdx.z * head_num + blockIdx.y) * seq_len + blockIdx.x) * seq_len +
                   blockDim.x * idx + threadIdx.x;

    out[input_offset] = static_cast<T>(data[idx] * s_rsum);
  }
}

template <typename T>
void mask_softmax_fprop(T* out, T* in, T* mask, int batch_size, int head_num, int seq_len,
                        float scalar, cudaStream_t stream) {
  dim3 grid(seq_len, head_num, batch_size);
  int block_len = max(32, (seq_len + 31) / 32 * 32);
  dim3 block(min(block_len, 1024));
  mask_softmax_fprop_kernel<<<grid, block, 0, stream>>>(out, in, mask, batch_size, head_num,
                                                        seq_len, scalar);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void __global__ mask_softmax_bprop_kernel(T* top, T* bottom, T* softmax, int m, int n,
                                          float scalar) {
  int offset = blockIdx.x * n;
  float grad_softmax = static_cast<float>(0.0f);
  __shared__ float grad_sum;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    grad_softmax += top[idx] * softmax[idx];
  }
  float tmp = blockReduceSum<T>(grad_softmax);
  if (threadIdx.x == 0) {
    grad_sum = tmp;
  }
  __syncthreads();

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    bottom[idx] = (softmax[idx] * top[idx] - softmax[idx] * grad_sum) * scalar;
  }
}

template <>
void __global__ mask_softmax_bprop_kernel(__half* top, __half* bottom, __half* softmax, int m,
                                          int n, float scalar) {
  int offset = blockIdx.x * n;
  float grad_softmax = static_cast<float>(0.0f);
  __shared__ __half grad_sum;

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    grad_softmax += static_cast<float>(top[idx] * softmax[idx]);
  }

  float tmp = blockReduceSum<float>(grad_softmax);
  if (threadIdx.x == 0) {
    grad_sum = static_cast<__half>(tmp);
  }

  __syncthreads();
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    __half tmp = __hsub(top[idx], grad_sum);
    bottom[idx] = __hmul(bottom[idx], tmp);
    bottom[idx] = __hdiv(bottom[idx], scalar);
  }
}

template <typename T>
void mask_softmax_bprop(T* top, T* bottom, T* softmax, int m, int n, float scalar,
                        cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  mask_softmax_bprop_kernel<<<grid, block, 0, stream>>>(top, bottom, softmax, m, n, scalar);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void MaskedSoftmaxLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  Tensor2<T>& in_tensor = in_tensors_[0];
  Tensor2<T>& mask_tensor = in_tensors_[1];
  Tensor2<T>& out_tensor = out_tensors_[0];
  const auto& in_tensor_dim = in_tensor.get_dimensions();

  mask_softmax_fprop(out_tensor.get_ptr(), in_tensor.get_ptr(), mask_tensor.get_ptr(),
                     in_tensor_dim[0], in_tensor_dim[1], in_tensor_dim[2], scalar_,
                     get_gpu().get_stream());
  HCTR_LIB_THROW(cudaMemcpyAsync((void*)softmax_out_.get_ptr(), (void*)out_tensor.get_ptr(),
                                 out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <>
void MaskedSoftmaxLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  Tensor2<__half>& in_tensor = in_tensors_[0];
  Tensor2<__half>& mask_tensor = in_tensors_[1];
  Tensor2<__half>& out_tensor = out_tensors_[0];
  const auto& in_tensor_dim = in_tensor.get_dimensions();
  mask_softmax_fprop(out_tensor.get_ptr(), in_tensor.get_ptr(), mask_tensor.get_ptr(),
                     in_tensor_dim[0], in_tensor_dim[1], in_tensor_dim[2], scalar_,
                     get_gpu().get_stream());
  HCTR_LIB_THROW(cudaMemcpyAsync((void*)softmax_out_.get_ptr(), (void*)out_tensor.get_ptr(),
                                 out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void MaskedSoftmaxLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  Tensor2<T>& bottom_tensor = in_tensors_[0];
  Tensor2<T>& top_tensor = out_tensors_[0];
  const auto& in_tensor_dim = bottom_tensor.get_dimensions();
  int hidden_size = in_tensor_dim[in_tensor_dim.size() - 1];
  int batch = bottom_tensor.get_num_elements() / hidden_size;

  mask_softmax_bprop(top_tensor.get_ptr(), bottom_tensor.get_ptr(), softmax_out_.get_ptr(), batch,
                     hidden_size, scalar_, get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <>
void MaskedSoftmaxLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());
  Tensor2<__half>& bottom_tensor = in_tensors_[0];
  Tensor2<__half>& top_tensor = out_tensors_[0];
  const auto& in_tensor_dim = bottom_tensor.get_dimensions();

  int hidden_size = in_tensor_dim[in_tensor_dim.size() - 1];
  int n_rows = bottom_tensor.get_num_elements() / hidden_size;

  mask_softmax_bprop(top_tensor.get_ptr(), bottom_tensor.get_ptr(), softmax_out_.get_ptr(), n_rows,
                     hidden_size, scalar_, get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class MaskedSoftmaxLayer<float>;
template class MaskedSoftmaxLayer<__half>;

}  // namespace HugeCTR

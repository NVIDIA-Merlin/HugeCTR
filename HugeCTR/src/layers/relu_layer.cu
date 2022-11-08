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

#include <cuda_fp16.h>

#include <algorithm>
#include <cuda/std/array>
#include <functional>
#include <include/utils.cuh>
#include <layers/element_wise_function.hpp>
#include <layers/relu_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/unary_op.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

struct alignas(8) half4 : public cuda::std::array<__half2, 2> {};

template <typename MainOp, typename Fallback>
__global__ void half4_relu_kernel(__half* __restrict__ out, const __half* __restrict__ in, int size,
                                  MainOp main_op, Fallback fallback) {
  const __half2 zero2 = TypeFunc<__half2>::zero();
  const half4 zero4 = {zero2, zero2};
  half4* out4 = reinterpret_cast<half4*>(out);
  const half4* in4 = reinterpret_cast<const half4*>(in);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size4 = size / 4;

  for (int i = tid; i < size4; i += stride) {
    main_op(in4, zero2, i, out4);
  }

  const __half zero = TypeFunc<__half>::zero();
  int rmdr_base = size4 * 4;

  for (int i = rmdr_base + tid; i < size; i += stride) {
    fallback(in, zero, i, out);
  }
}

}  // namespace

template <typename T>
ReluLayer<T>::ReluLayer(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                        const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);
}

template <typename T>
void ReluLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  int len = in_tensors_[0].get_num_elements();

  auto fop = [] __device__(T in) { return (in > T(0)) ? in : T(0); };

  MLCommon::LinAlg::unaryOp(out_tensors_[0].get_ptr(), in_tensors_[0].get_ptr(), len, fop,
                            get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void ReluLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  int len = in_tensors_[0].get_num_elements();

  auto bop = [] __device__(T d_out, T d_in) { return (d_in > T(0)) ? d_out : T(0); };

  MLCommon::LinAlg::binaryOp(in_tensors_[0].get_ptr(), out_tensors_[0].get_ptr(),
                             in_tensors_[0].get_ptr(), len, bop, get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

ReluLayer<__half>::ReluLayer(const Tensor2<__half>& bottom_tensor,
                             const Tensor2<__half>& top_tensor,
                             const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  assert(get_size_from_dims(bottom_tensor.get_dimensions()) ==
         get_size_from_dims(top_tensor.get_dimensions()));

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
}

void ReluLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const size_t BLOCK_DIM = 1024;

  const size_t size = bottom_tensor_.get_num_elements();
  const size_t grid_dim = get_gpu().get_sm_count() * 4;

  half4_relu_kernel<<<grid_dim, BLOCK_DIM, 0, get_gpu().get_stream()>>>(
      top_tensor_.get_ptr(), bottom_tensor_.get_ptr(), size,
      [] __device__(const half4* in4, const __half2 zero2, int i, half4* out4) {
        const int2 hack = reinterpret_cast<const int2*>(in4)[i];
        half4 t = *reinterpret_cast<const half4*>(&hack);

        const half4 mask = {__hgt2(t[0], zero2), __hgt2(t[1], zero2)};
        const half4 res = {__hmul2(t[0], mask[0]), __hmul2(t[1], mask[1])};

        reinterpret_cast<int2*>(out4)[i] = *reinterpret_cast<const int2*>(&res);
      },
      [] __device__(const __half* in, const __half zero, int i, __half* out) {
        __half t = __ldg(in + i);
        __half mask = __hgt(t, zero);
        out[i] = __hmul(t, mask);
      });

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

void ReluLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());

  const size_t BLOCK_DIM = 1024;

  const size_t size = bottom_tensor_.get_num_elements();
  const size_t grid_dim = get_gpu().get_sm_count() * 4;
  half4_relu_kernel<<<grid_dim, BLOCK_DIM, 0, get_gpu().get_stream()>>>(
      bottom_tensor_.get_ptr(), top_tensor_.get_ptr(), size,
      [] __device__(const half4* in4, const __half2 zero2, int i, half4* out4) {
        const int2 t_hack = reinterpret_cast<const int2*>(out4)[i];
        const half4 t = *reinterpret_cast<const half4*>(&t_hack);

        const half4 mask = {__hgt2(t[0], zero2), __hgt2(t[1], zero2)};

        const int2 t2_hack = reinterpret_cast<const int2*>(in4)[i];
        const half4 t2 = *reinterpret_cast<const half4*>(&t2_hack);

        const half4 res = {__hmul2(t2[0], mask[0]), __hmul2(t2[1], mask[1])};

        reinterpret_cast<int2*>(out4)[i] = *reinterpret_cast<const int2*>(&res);
      },
      [] __device__(const __half* in, const __half zero, int i, __half* out) {
        __half t = out[i];
        __half mask = __hgt(t, zero);
        out[i] = __hmul(__ldg(in + i), mask);
      });

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class ReluLayer<float>;
template class ReluLayer<__half>;

}  // namespace HugeCTR

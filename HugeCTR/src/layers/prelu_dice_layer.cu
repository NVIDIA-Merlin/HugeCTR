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
#include <cuda_utils.cuh>
#include <functional>
#include <include/utils.cuh>
#include <layers/element_wise_function.hpp>
#include <layers/prelu_dice_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/reduce.cuh>
#include <linalg/unary_op.cuh>
#include <utils.hpp>

namespace HugeCTR {

template <typename T>
PRelu_Dice_Layer<T>::PRelu_Dice_Layer(
    const Tensor2<T> &in_tensor, const Tensor2<T> &out_tensor,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &blobs_buff, T alpha, T epsilon,
    const std::shared_ptr<GPUResource> &gpu_resource)
    : Layer(gpu_resource), alpha_(alpha), epsilon_(epsilon) {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);
  len = in_tensors_[0].get_num_elements();
  batchsize_ = in_tensor.get_dimensions()[0];
  hiddensize_ = len / batchsize_;
  blobs_buff->reserve({hiddensize_}, &E_x);
  blobs_buff->reserve({hiddensize_}, &Var_x);
  blobs_buff->reserve({hiddensize_}, &E_x2);
}

template <typename T>
void __global__ Dice_fprop_kernel(T *out, T *in, T *E_x, T *Var_x, T alpha, T epsilon, int m,
                                  int n) {
  int offset = blockIdx.x * n;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    T ps = 1 / (1 + expf((E_x[tid] - in[offset + tid]) / sqrt(Var_x[tid] + epsilon)));
    out[offset + tid] = ps * in[offset + tid] + (1 - ps) * alpha * in[offset + tid];
  }
}

template <typename T>
void Dice_fprop(T *out, T *in, T *E_x, T *Var_x, T alpha, T epsilon, int m, int n,
                cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  Dice_fprop_kernel<<<grid, block, 0, stream>>>(out, in, E_x, Var_x, alpha, epsilon, m, n);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void __global__ Dice_bprop_kernel(T *top, T *bottom, T *E_x, T *Var_x, T alpha, T epsilon, int m,
                                  int n) {
  int offset = blockIdx.x * n;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    T Ex = E_x[tid];
    T Vx = Var_x[tid];
    T s = bottom[offset + tid];
    T divide_zero = Vx + epsilon < 1e-8 ? 0.0 : (1.0 / sqrt(Vx + epsilon));
    T divide_zero_pow = Vx + epsilon < 1e-8 ? 0.0 : (1.0 / sqrt(pow(Vx + epsilon, 3.0)));

    T ys_s = (T(1.0 / m) - 1) * divide_zero - T(1.0 / m) * (Ex - s) * divide_zero_pow * (s - Ex);
    T ys = (Ex - s) * divide_zero;
    T ps_s = -1.0 * expf(ys) * ys_s * (1.0 / pow(1.0 + expf(ys), 2.0));
    T ps = 1.0 / (1 + expf(ys));
    bottom[offset + tid] = ((ps_s * s + ps) * (1.0 - alpha) + alpha) * top[offset + tid];
  }
}

template <typename T>
void Dice_bprop(T *top, T *bottom, T *E_x, T *Var_x, T alpha, T epsilon, int m, int n,
                cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  Dice_bprop_kernel<<<grid, block, 0, stream>>>(top, bottom, E_x, Var_x, alpha, epsilon, m, n);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void PRelu_Dice_Layer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  Tensor2<T> &in_tensor = in_tensors_[0];
  Tensor2<T> &out_tensor = out_tensors_[0];
  const auto &in_tensor_dim = in_tensor.get_dimensions();

  T alpha = alpha_;
  T epsilon = epsilon_;
  int batchsize = batchsize_;
  int hiddensize = hiddensize_;

  // Get mean of each batch.
  MLCommon::LinAlg::reduce(
      E_x.get_ptr(), in_tensor.get_ptr(), hiddensize, batchsize, T(0), true, false,
      get_gpu().get_stream(), false, [] __device__(T in, int i) { return in; },
      [] __device__(T a, T b) { return a + b; },
      [batchsize] __device__(T out) { return out / batchsize; });

  // Get Variance of each batch. Var_x = E(x^2) - E(x)^2;
  // E(x^2);
  MLCommon::LinAlg::reduce(
      E_x2.get_ptr(), in_tensor.get_ptr(), hiddensize, batchsize, T(0), true, false,
      get_gpu().get_stream(), false, [] __device__(T in, int i) { return pow(in, 2.0); },
      [] __device__(T a, T b) { return a + b; },
      [batchsize] __device__(T out) { return out / batchsize; });
  // E(x)^2;
  MLCommon::LinAlg::unaryOp(
      Var_x.get_ptr(), E_x.get_ptr(), hiddensize, [] __device__(T in) { return pow(in, 2.0); },
      get_gpu().get_stream());
  // Var_x = E(x^2) - E(x)^2;
  MLCommon::LinAlg::binaryOp(
      Var_x.get_ptr(), E_x2.get_ptr(), Var_x.get_ptr(), hiddensize,
      [] __device__(T a, T b) { return a - b; }, get_gpu().get_stream());

  Dice_fprop(out_tensor.get_ptr(), in_tensor.get_ptr(), E_x.get_ptr(), Var_x.get_ptr(), alpha,
             epsilon, in_tensor_dim[0], in_tensor_dim[1], get_gpu().get_stream());
}

template <typename T>
void PRelu_Dice_Layer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  Tensor2<T> &bottom_tensor = in_tensors_[0];
  Tensor2<T> &top_tensor = out_tensors_[0];
  const auto &bottom_tensor_dim = bottom_tensor.get_dimensions();
  T alpha = alpha_;
  T epsilon = epsilon_;

  Dice_bprop(top_tensor.get_ptr(), bottom_tensor.get_ptr(), E_x.get_ptr(), Var_x.get_ptr(), alpha,
             epsilon, bottom_tensor_dim[0], bottom_tensor_dim[1], get_gpu().get_stream());
}

template class PRelu_Dice_Layer<float>;
// template class PRelu_Dice_Layer<__half>;

}  // namespace HugeCTR

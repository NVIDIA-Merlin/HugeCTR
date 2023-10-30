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
#include <layers/prelu_dice_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/reduce.cuh>
#include <linalg/unary_op.cuh>
#include <utils.hpp>

namespace HugeCTR {

template <typename T>
PRelu_Dice_Layer<T>::PRelu_Dice_Layer(const core23::Tensor &input_tensor,
                                      const core23::Tensor &output_tensor, T alpha, T epsilon,
                                      const std::shared_ptr<GPUResource> &gpu_resource)
    : Layer({input_tensor}, {output_tensor}, gpu_resource), alpha_(alpha), epsilon_(epsilon) {
  assert(input_tensor.num_elements() == output_tensor.num_elements());

  len = input_tensors_[0].num_elements();
  batchsize_ = input_tensor.shape().size(0);
  hiddensize_ = len / batchsize_;
  E_x_ = core23::Tensor({(int64_t)hiddensize_}, core23::DataType(core23::ToScalarType<T>::value));
  Var_x_ = core23::Tensor({(int64_t)hiddensize_}, core23::DataType(core23::ToScalarType<T>::value));
  E_x2_ = core23::Tensor({(int64_t)hiddensize_}, core23::DataType(core23::ToScalarType<T>::value));
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
}

template <typename T>
void PRelu_Dice_Layer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  core23::Tensor &input_tensor = input_tensors_[0];
  core23::Tensor &output_tensor = output_tensors_[0];
  const auto &input_tensor_shape = input_tensor.shape();

  T alpha = alpha_;
  T epsilon = epsilon_;
  int batchsize = batchsize_;
  int hiddensize = hiddensize_;

  // Get mean of each batch.
  MLCommon::LinAlg::reduce(
      E_x_.data<T>(), input_tensor.data<T>(), hiddensize, batchsize, T(0), true, false,
      get_gpu().get_stream(), false, [] __device__(T in, int i) { return in; },
      [] __device__(T a, T b) { return a + b; },
      [batchsize] __device__(T out) { return out / batchsize; });

  // Get Variance of each batch. Var_x = E(x^2) - E(x)^2;
  // E(x^2);
  MLCommon::LinAlg::reduce(
      E_x2_.data<T>(), input_tensor.data<T>(), hiddensize, batchsize, T(0), true, false,
      get_gpu().get_stream(), false, [] __device__(T in, int i) { return pow(in, 2.0); },
      [] __device__(T a, T b) { return a + b; },
      [batchsize] __device__(T out) { return out / batchsize; });
  // E(x)^2;
  MLCommon::LinAlg::unaryOp(
      Var_x_.data<T>(), E_x_.data<T>(), hiddensize, [] __device__(T in) { return pow(in, 2.0); },
      get_gpu().get_stream());
  // Var_x = E(x^2) - E(x)^2;
  MLCommon::LinAlg::binaryOp(
      Var_x_.data<T>(), E_x2_.data<T>(), Var_x_.data<T>(), hiddensize,
      [] __device__(T a, T b) { return a - b; }, get_gpu().get_stream());

  Dice_fprop(output_tensor.data<T>(), input_tensor.data<T>(), E_x_.data<T>(), Var_x_.data<T>(),
             alpha, epsilon, input_tensor_shape.size(0), input_tensor_shape.size(1),
             get_gpu().get_stream());
}

template <typename T>
void PRelu_Dice_Layer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  core23::Tensor &bottom_tensor = input_tensors_[0];
  core23::Tensor &top_tensor = output_tensors_[0];
  const auto &bottom_tensor_shape = bottom_tensor.shape();
  T alpha = alpha_;
  T epsilon = epsilon_;

  Dice_bprop(top_tensor.data<T>(), bottom_tensor.data<T>(), E_x_.data<T>(), Var_x_.data<T>(), alpha,
             epsilon, bottom_tensor_shape.size(0), bottom_tensor_shape.size(1),
             get_gpu().get_stream());
}

template class PRelu_Dice_Layer<float>;
// template class PRelu_Dice_Layer<__half>;

}  // namespace HugeCTR

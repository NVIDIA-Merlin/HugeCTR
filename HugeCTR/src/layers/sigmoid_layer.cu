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
 #include <layers/sigmoid_layer.hpp>
 #include <linalg/binary_op.cuh>
 #include <linalg/unary_op.cuh>
 #include <utils.hpp>
 
 #ifndef NDEBUG
 #include <iostream>
 #endif

 namespace HugeCTR {
 template <typename T>
 __device__ T exponential(T in) { return exp(in); }
 template __device__ float exponential<float>(float);
 template <> __device__ __half exponential(__half in) { return hexp(in); }

 template <typename T>
 SigmoidLayer<T>::SigmoidLayer(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                         const std::shared_ptr<GPUResource>& gpu_resource)
     : Layer(gpu_resource) {
   assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());
   assert(in_tensor.get_num_elements() % 2 == 0);
 
   in_tensors_.push_back(in_tensor);
   out_tensors_.push_back(out_tensor);
 }
 
 template <typename T>
 void SigmoidLayer<T>::fprop(bool is_train) {
   CudaDeviceContext context(get_device_id());
 
   int len = in_tensors_[0].get_num_elements();
 
   auto fop = [] __device__(T in) { return T(1) / ( T(1)+exponential(-in) ); };
 
   MLCommon::LinAlg::unaryOp(out_tensors_[0].get_ptr(), in_tensors_[0].get_ptr(), len, fop,
                             get_gpu().get_stream());
 
 #ifndef NDEBUG
   cudaDeviceSynchronize();
   CK_CUDA_THROW_(cudaGetLastError());
 #endif
 }
 
 template <typename T>
 void SigmoidLayer<T>::bprop() {
   CudaDeviceContext context(get_device_id());
 
   int len = in_tensors_[0].get_num_elements();
 
   auto bop = [] __device__(T d_out, T d_in) {
     T y = T(1) / ( T(1)+exponential(-d_in) );
     return d_out*y*(T(1)-y);
   };
 
   MLCommon::LinAlg::binaryOp(in_tensors_[0].get_ptr(), out_tensors_[0].get_ptr(),
                              in_tensors_[0].get_ptr(), len, bop, get_gpu().get_stream());
 
 #ifndef NDEBUG
   cudaDeviceSynchronize();
   CK_CUDA_THROW_(cudaGetLastError());
 #endif
 }
 
 template class SigmoidLayer<float>;
 template class SigmoidLayer<__half>;
 
 }  // namespace HugeCTR
 
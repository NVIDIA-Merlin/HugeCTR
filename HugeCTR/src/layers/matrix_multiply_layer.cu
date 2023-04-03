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
#include <layers/matrix_multiply_layer.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {
template <typename T>
MatrixMultiplyLayer<T>::MatrixMultiplyLayer(const std::vector<core23::Tensor>& input_tensors,
                                            core23::Tensor& output_tensor,
                                            const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({}, {}, gpu_resource) {
  try {
    num_ = input_tensors.size();

    // error input checking
    dims_ = input_tensors[0].shape().dims();
    if (num_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "MatrixMultiplyLayer needs at least 2 input tensors");
    }
    if (input_tensors[1].shape().dims() != (int64_t)dims_) {
      HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
    }
    if (input_tensors[1].shape().size(dims_ - 2) != input_tensors[0].shape().size(dims_ - 1)) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "The last two dimension of the input tensors should be m x n, n x k");
    }

    for (size_t i = 0; i < num_; i++) {
      input_tensors_.push_back(input_tensors[i]);
    }

    int64_t m = input_tensors[0].shape().size(dims_ - 2);
    int64_t k = input_tensors[1].shape().size(dims_ - 1);

    core23::TensorParams out_params = input_tensors[0].my_params();
    if (dims_ == 2) {
      std::vector<int64_t> out_shape = {m, k};
      output_tensor = core23::Tensor(out_params.shape(out_shape));
    } else if (dims_ == 3) {  // dims_ == 3
      if (input_tensors[0].shape().size(0) != input_tensors[1].shape().size(0)) {
        HCTR_OWN_THROW(Error_t::WrongInput, "3D input tensors must have the same batch size");
      }
      int64_t b = input_tensors[0].shape().size(0);
      std::vector<int64_t> out_shape = {b, m, k};
      output_tensor = core23::Tensor(out_params.shape(out_shape));
    } else if (dims_ == 4) {
      if (input_tensors[0].shape().size(0) != input_tensors[1].shape().size(0)) {
        HCTR_OWN_THROW(Error_t::WrongInput, "4D input tensors must have the same batch size");
      }
      if (input_tensors[0].shape().size(1) != input_tensors[1].shape().size(1)) {
        HCTR_OWN_THROW(Error_t::WrongInput, "4D input tensors must have the same second dim");
      }
      int64_t b = input_tensors[0].shape().size(0);
      int64_t num_head = input_tensors[0].shape().size(1);
      std::vector<int64_t> out_shape = {b, num_head, m, k};
      output_tensor = core23::Tensor(out_params.shape(out_shape));
    }

    output_tensors_.push_back(output_tensor);

    fprop_inputA_tensor23_ = core23::Tensor(out_params);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
MatrixMultiplyLayer<T>::MatrixMultiplyLayer(
    const Tensors2<T>& in_tensors, Tensor2<T>& out_tensor,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    num_ = in_tensors.size();

    // error input checking
    dims_ = in_tensors[0].get_dimensions().size();
    if (num_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "MatrixMultiplyLayer needs at least 2 input tensors");
    }
    if (in_tensors[1].get_dimensions().size() != dims_) {
      HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
    }
    if (in_tensors[1].get_dimensions()[dims_ - 2] != in_tensors[0].get_dimensions()[dims_ - 1]) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "The last two dimension of the input tensors should be m x n, n x k");
    }

    for (size_t i = 0; i < num_; i++) {
      in_tensors_.push_back(in_tensors[i]);
    }

    size_t m = in_tensors[0].get_dimensions()[dims_ - 2];
    size_t k = in_tensors[1].get_dimensions()[dims_ - 1];

    if (dims_ == 2) {
      std::vector<size_t> out_dim = {m, k};
      blobs_buff->reserve(out_dim, &out_tensor);
    } else if (dims_ == 3) {  // dims_ == 3
      if (in_tensors[0].get_dimensions()[0] != in_tensors[1].get_dimensions()[0]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "3D input tensors must have the same batch size");
      }
      size_t b = in_tensors[0].get_dimensions()[0];
      std::vector<size_t> out_dim = {b, m, k};
      blobs_buff->reserve(out_dim, &out_tensor);
    } else if (dims_ == 4) {
      if (in_tensors[0].get_dimensions()[0] != in_tensors[1].get_dimensions()[0]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "4D input tensors must have the same batch size");
      }
      if (in_tensors[0].get_dimensions()[1] != in_tensors[1].get_dimensions()[1]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "4D input tensors must have the same second dim");
      }
      size_t b = in_tensors[0].get_dimensions()[0];
      size_t num_head = in_tensors[0].get_dimensions()[1];
      std::vector<size_t> out_dim = {b, num_head, m, k};
      blobs_buff->reserve(out_dim, &out_tensor);
    }

    out_tensors_.push_back(out_tensor);

    blobs_buff->reserve(in_tensors[0].get_dimensions(), &fprop_inputA_);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void MatrixMultiplyLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  if (input_tensors_.empty()) {
    T* in1 = in_tensors_[0].get_ptr();
    T* in2 = in_tensors_[1].get_ptr();
    T* out = out_tensors_[0].get_ptr();

    const auto& in_tensor_dim = in_tensors_[0].get_dimensions();
    const auto& out_tensor_dim = out_tensors_[0].get_dimensions();

    size_t m, n, k, b = 1;

    b = dims_ == 3 ? in_tensor_dim[0] : 1;
    b = dims_ == 4 ? in_tensor_dim[0] * in_tensor_dim[1] : b;
    m = in_tensor_dim[dims_ - 2];
    n = in_tensor_dim[dims_ - 1];
    k = out_tensor_dim[dims_ - 1];
    float alpha = 1.0f, beta = 0.0f;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    for (size_t i = 0; i < b; i++) {
      T* cur_in1 = in1 + i * m * n;
      T* cur_in2 = in2 + i * n * k;
      T* cur_out = out + i * m * k;
      HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, k, m, n,
                                  &alpha, cur_in2, CUDA_R_32F, k, cur_in1, CUDA_R_32F, n, &beta,
                                  cur_out, CUDA_R_32F, k, compute_type, CUBLAS_GEMM_DEFAULT));
    }

    HCTR_LIB_THROW(cudaMemcpyAsync((void*)fprop_inputA_.get_ptr(), (void*)in1,
                                   in_tensors_[0].get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                   get_gpu().get_stream()));
  } else {
    T* in1 = input_tensors_[0].data<T>();
    T* in2 = input_tensors_[1].data<T>();
    T* out = output_tensors_[0].data<T>();

    const auto& input_tensor_shape = input_tensors_[0].shape();
    const auto& output_tensor_shape = output_tensors_[0].shape();

    int64_t m, n, k, b = 1;

    b = dims_ == 3 ? input_tensor_shape.size(0) : 1;
    b = dims_ == 4 ? input_tensor_shape.size(0) * input_tensor_shape.size(1) : b;
    m = input_tensor_shape.size(dims_ - 2);
    n = input_tensor_shape.size(dims_ - 1);
    k = output_tensor_shape.size(dims_ - 1);
    float alpha = 1.0f, beta = 0.0f;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    for (auto i = 0; i < b; i++) {
      T* cur_in1 = in1 + i * m * n;
      T* cur_in2 = in2 + i * n * k;
      T* cur_out = out + i * m * k;
      HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, k, m, n,
                                  &alpha, cur_in2, CUDA_R_32F, k, cur_in1, CUDA_R_32F, n, &beta,
                                  cur_out, CUDA_R_32F, k, compute_type, CUBLAS_GEMM_DEFAULT));
    }

    HCTR_LIB_THROW(cudaMemcpyAsync(fprop_inputA_tensor23_.data(), (void*)in1,
                                   input_tensors_[0].num_bytes(), cudaMemcpyDeviceToDevice,
                                   get_gpu().get_stream()));
  }
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void MatrixMultiplyLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  if (input_tensors_.empty()) {
    T* in1 = in_tensors_[0].get_ptr();
    T* in2 = in_tensors_[1].get_ptr();
    T* out = out_tensors_[0].get_ptr();

    const auto& in_tensor_dim = in_tensors_[0].get_dimensions();
    const auto& out_tensor_dim = out_tensors_[0].get_dimensions();

    size_t m, n, k, b = 1;

    b = dims_ == 3 ? in_tensor_dim[0] : 1;
    b = dims_ == 4 ? in_tensor_dim[0] * in_tensor_dim[1] : b;
    m = in_tensor_dim[dims_ - 2];
    n = in_tensor_dim[dims_ - 1];
    k = out_tensor_dim[dims_ - 1];
    float alpha = 1.0f, beta = 0.0f;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    for (size_t i = 0; i < b; i++) {
      T* cur_in1 = in1 + i * m * n;
      T* cur_in2 = in2 + i * n * k;
      T* cur_out = out + i * m * k;
      // gradient respect to A
      HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                  &alpha, cur_in2, CUDA_R_32F, k, cur_out, CUDA_R_32F, k, &beta,
                                  cur_in1, CUDA_R_32F, n, compute_type, CUBLAS_GEMM_DEFAULT));

      cur_in1 = fprop_inputA_.get_ptr() + i * m * n;
      // gradient respect to B
      HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, k, n, m,
                                  &alpha, cur_out, CUDA_R_32F, k, cur_in1, CUDA_R_32F, n, &beta,
                                  cur_in2, CUDA_R_32F, k, compute_type, CUBLAS_GEMM_DEFAULT));
    }
  } else {
    T* in1 = input_tensors_[0].data<T>();
    T* in2 = input_tensors_[1].data<T>();
    T* out = output_tensors_[0].data<T>();

    const auto& input_tensor_shape = input_tensors_[0].shape();
    const auto& output_tensor_shape = output_tensors_[0].shape();

    int64_t m, n, k, b = 1;

    b = dims_ == 3 ? input_tensor_shape.size(0) : 1;
    b = dims_ == 4 ? output_tensor_shape.size(0) * input_tensor_shape.size(1) : b;
    m = input_tensor_shape.size(dims_ - 2);
    n = input_tensor_shape.size(dims_ - 1);
    k = output_tensor_shape.size(dims_ - 1);
    float alpha = 1.0f, beta = 0.0f;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    for (auto i = 0; i < b; i++) {
      T* cur_in1 = in1 + i * m * n;
      T* cur_in2 = in2 + i * n * k;
      T* cur_out = out + i * m * k;
      // gradient respect to A
      HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                  &alpha, cur_in2, CUDA_R_32F, k, cur_out, CUDA_R_32F, k, &beta,
                                  cur_in1, CUDA_R_32F, n, compute_type, CUBLAS_GEMM_DEFAULT));

      cur_in1 = fprop_inputA_tensor23_.data<T>() + i * m * n;
      // gradient respect to B
      HCTR_LIB_THROW(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, k, n, m,
                                  &alpha, cur_out, CUDA_R_32F, k, cur_in1, CUDA_R_32F, n, &beta,
                                  cur_in2, CUDA_R_32F, k, compute_type, CUBLAS_GEMM_DEFAULT));
    }
  }
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class MatrixMultiplyLayer<float>;

}  // namespace HugeCTR

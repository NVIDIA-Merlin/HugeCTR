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
    dims_lhs_ = input_tensors[0].shape().dims();
    dims_rhs_ = input_tensors[1].shape().dims();
    const auto& dim_lhs = input_tensors[0].shape();
    const auto& dim_rhs = input_tensors[1].shape();

    if (num_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "MatrixMultiplyLayer needs at least 2 input tensors");
    }
    if (dims_lhs_ < 2 || dims_rhs_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "MatrixMultiplyLayer inputs should have at least 2 dims");
    }
    if (dims_lhs_ == 2 && dims_rhs_ == 3) {
      HCTR_CHECK_HINT(dim_rhs[0] == dim_lhs[1], "MatrixMultiplyLayer 2Dx3D invalid shape");
    } else if (dims_lhs_ != dims_rhs_) {
      HCTR_OWN_THROW(Error_t::WrongInput, "MatrixMultiplyLayer invalid input shape");
    }

    if (dim_rhs[dims_lhs_ - 2] != dim_lhs[dims_lhs_ - 1]) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "The last two dimension of the input tensors should be m x n, n x k");
    }

    for (size_t i = 0; i < num_; i++) {
      input_tensors_.push_back(input_tensors[i]);
    }

    int64_t m = input_tensors[0].shape().size(dims_lhs_ - 2);
    int64_t k = input_tensors[1].shape().size(dims_lhs_ - 1);
    if (dims_lhs_ == 2 && dims_rhs_ == 3) {
      k = dim_rhs[1] * dim_rhs[2];
    }
    core23::TensorParams out_params = input_tensors[0].my_params();

    if (dims_lhs_ == 2) {
      if (dims_rhs_ == 2) {
        std::vector<int64_t> out_shape = {m, k};
        output_tensor = core23::Tensor(out_params.shape(out_shape));
      } else {
        std::vector<int64_t> out_shape = {m, dim_rhs[1], dim_rhs[2]};
        output_tensor = core23::Tensor(out_params.shape(out_shape));
      }
    } else if (dims_lhs_ == 3) {
      if (dim_lhs[0] != dim_rhs[0]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "3D input tensors must have the same batch size");
      }
      int64_t b = dim_lhs[0];
      std::vector<int64_t> out_shape = {b, m, k};
      output_tensor = core23::Tensor(out_params.shape(out_shape));
    } else if (dims_lhs_ == 4) {
      if (dim_lhs[0] != dim_rhs[0]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "4D input tensors must have the same batch size");
      }
      if (dim_lhs[1] != dim_rhs[1]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "4D input tensors must have the same second dim");
      }
      int64_t b = dim_lhs[0];
      int64_t num_head = dim_lhs[1];
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

/*
  1. 2D: (m, n), (n, k)  and output: (m, k)
  2. 3D: (batch_size, m, n), (batch_size, n, k) and output will: (batch_size, m, k)
  3. 2D x 3D: (m, n) , (n, g, h), and output will be:  (m, g, h)
*/
template <typename T>
MatrixMultiplyLayer<T>::MatrixMultiplyLayer(
    const Tensors2<T>& in_tensors, Tensor2<T>& out_tensor,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    num_ = in_tensors.size();

    // error input checking
    dims_lhs_ = in_tensors[0].get_dimensions().size();
    dims_rhs_ = in_tensors[1].get_dimensions().size();
    const auto& dim_lhs = in_tensors[0].get_dimensions();
    const auto& dim_rhs = in_tensors[1].get_dimensions();

    if (num_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "MatrixMultiplyLayer needs at least 2 input tensors");
    }
    if (dims_lhs_ < 2 || dims_rhs_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "MatrixMultiplyLayer inputs should have at least 2 dims");
    }

    if (dims_lhs_ == 2 && dims_rhs_ == 3) {
      HCTR_CHECK_HINT(dim_rhs[0] == dim_lhs[1], "MatrixMultiplyLayer 2Dx3D invalid shape");
    } else if (dims_lhs_ != dims_rhs_) {
      HCTR_OWN_THROW(Error_t::WrongInput, "MatrixMultiplyLayer invalid input shape");
    }

    if (dim_rhs[dims_lhs_ - 2] != dim_lhs[dims_lhs_ - 1]) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "The last two dimension of the input tensors should be m x n, n x k");
    }

    for (size_t i = 0; i < num_; i++) {
      in_tensors_.push_back(in_tensors[i]);
    }

    size_t m = dim_lhs[dims_lhs_ - 2];
    size_t k = dim_rhs[dims_rhs_ - 1];
    if (dims_lhs_ == 2 && dims_rhs_ == 3) {
      k = dim_rhs[1] * dim_rhs[2];
    }
    if (dims_lhs_ == 2) {
      if (dims_rhs_ == 2) {
        std::vector<size_t> out_dim = {m, k};
        blobs_buff->reserve(out_dim, &out_tensor);
      } else {
        std::vector<size_t> out_dim = {m, dim_rhs[1], dim_rhs[2]};
        blobs_buff->reserve(out_dim, &out_tensor);
      }
    } else if (dims_lhs_ == 3) {
      if (dim_lhs[0] != dim_rhs[0]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "3D input tensors must have the same batch size");
      }
      size_t b = dim_lhs[0];
      std::vector<size_t> out_dim = {b, m, k};
      blobs_buff->reserve(out_dim, &out_tensor);
    } else if (dims_lhs_ == 4) {
      if (dim_lhs[0] != dim_rhs[0]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "4D input tensors must have the same batch size");
      }
      if (dim_lhs[1] != dim_rhs[1]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "4D input tensors must have the same second dim");
      }
      size_t b = dim_lhs[0];
      size_t num_head = dim_lhs[1];
      std::vector<size_t> out_dim = {b, num_head, m, k};
      blobs_buff->reserve(out_dim, &out_tensor);
    }

    out_tensors_.push_back(out_tensor);

    blobs_buff->reserve(dim_lhs, &fprop_inputA_);

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

    b = dims_lhs_ == 3 ? in_tensor_dim[0] : 1;
    b = dims_lhs_ == 4 ? in_tensor_dim[0] * in_tensor_dim[1] : b;
    m = in_tensor_dim[dims_lhs_ - 2];
    n = in_tensor_dim[dims_lhs_ - 1];
    k = out_tensor_dim[dims_lhs_ - 1];
    if (dims_lhs_ == 2 && dims_rhs_ == 3) {
      k = out_tensor_dim[1] * out_tensor_dim[2];
    }
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

    b = dims_lhs_ == 3 ? input_tensor_shape.size(0) : 1;
    b = dims_lhs_ == 4 ? input_tensor_shape.size(0) * input_tensor_shape.size(1) : b;
    m = input_tensor_shape.size(dims_lhs_ - 2);
    n = input_tensor_shape.size(dims_lhs_ - 1);
    k = output_tensor_shape.size(dims_lhs_ - 1);
    if (dims_lhs_ == 2 && dims_rhs_ == 3) {
      k = output_tensor_shape[1] * output_tensor_shape[2];
    }
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

    b = dims_lhs_ == 3 ? in_tensor_dim[0] : 1;
    b = dims_lhs_ == 4 ? in_tensor_dim[0] * in_tensor_dim[1] : b;
    m = in_tensor_dim[dims_lhs_ - 2];
    n = in_tensor_dim[dims_lhs_ - 1];
    k = out_tensor_dim[dims_lhs_ - 1];
    if (dims_lhs_ == 2 && dims_rhs_ == 3) {
      k = out_tensor_dim[1] * out_tensor_dim[2];
    }
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

    b = dims_lhs_ == 3 ? input_tensor_shape[0] : 1;
    b = dims_lhs_ == 4 ? input_tensor_shape[0] * input_tensor_shape[1] : b;
    m = input_tensor_shape[dims_lhs_ - 2];
    n = input_tensor_shape[dims_lhs_ - 1];
    k = output_tensor_shape[dims_lhs_ - 1];
    if (dims_lhs_ == 2 && dims_rhs_ == 3) {
      k = output_tensor_shape[1] * output_tensor_shape[2];
    }
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
}

template class MatrixMultiplyLayer<float>;

}  // namespace HugeCTR

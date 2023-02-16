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

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <core23/tensor_container.hpp>
#include <data_simulator.hpp>
#include <layers/multi_cross_layer.hpp>
#include <memory>
#include <numeric>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

template <typename T>
class MultiCrossLayerTest {
 private:
  const float eps = 1;
  const size_t batchsize_;
  const size_t w_;
  const int layers_;
  const size_t projection_dim_;

  core23::Tensor d_input_;
  core23::Tensor d_output_;

  std::vector<T> h_input_;
  std::vector<T> h_input_grad_;
  std::vector<T> h_output_grad_;
  std::vector<std::vector<T>> XUs;
  std::vector<std::vector<T>> h_kernels_;  // weight 3D matrix
  std::vector<std::vector<T>> h_biases_;   // bias

  std::vector<std::vector<T>> h_outputs_;
  std::vector<std::vector<T>> h_hiddens_;

  std::vector<std::vector<T>> h_kernel_grads_;
  std::vector<std::vector<T>> h_bias_grads_;

  std::shared_ptr<Core23TempMultiCrossLayer<T>> layer_;
  test::GaussianDataSimulator data_sim_;

  std::vector<T*> weights_ptrs_;
  std::vector<T*> wgrads_ptrs_;

  constexpr std::vector<float> normal_params_() const noexcept {
    if (std::is_same<T, __half>::value) {
      return {0.0f, 0.125f, -2.f, 2.f};
    }
    return {0.0f, 1.0f, -2.f, 2.f};
  }

  constexpr T eps_() const noexcept {
    if (std::is_same<T, __half>::value) {
      return 2.0f;
    }
    return 1.0f;
  }

  void reset_forward_() {
    data_sim_.fill(h_input_.data(), batchsize_ * w_);
    std::transform(h_input_.begin(), h_input_.end(), h_input_.begin(),
                   [](T data) { return std::min((T)0.09, std::max(data, T(-0.09))); });
    for (auto& a : h_kernels_) {
      data_sim_.fill(a.data(), a.size());
      // double sum = std::accumulate(a.begin(), a.end(), 0.0);
      // double mean = sum / a.size();
      // std::cout<<" weight min "<<*std::min_element(a.begin(), a.end())<<" max
      // "<<*std::max_element(a.begin(), a.end())<<" mean "<< mean<<std::endl;
      std::transform(a.begin(), a.end(), a.begin(),
                     [](T data) { return std::min((T)1.0, std::max(data, T(-1.0))); });
      // std::cout<<" after transform weight min "<<*std::min_element(a.begin(), a.end())<<" max
      // "<<*std::max_element(a.begin(), a.end())<<" mean "<< mean<<std::endl;
    }
    for (auto& a : h_biases_) {
      data_sim_.fill(a.data(), w_);
      std::transform(a.begin(), a.end(), a.begin(),
                     [](T data) { return std::min((T)0.01, std::max(data, T(-0.01))); });
    }

    HCTR_LIB_THROW(
        cudaMemcpy(d_input_.data(), h_input_.data(), d_input_.num_bytes(), cudaMemcpyHostToDevice));

    int ptr_idx = 0;
    T* p;
    for (int i = 0; i < layers_; i++) {
      if (this->projection_dim_) {
        p = weights_ptrs_[ptr_idx++];
        HCTR_LIB_THROW(cudaMemcpy(p, h_kernels_[2 * i].data(),
                                  w_ * sizeof(T) * this->projection_dim_, cudaMemcpyHostToDevice));
        p = weights_ptrs_[ptr_idx++];
        HCTR_LIB_THROW(cudaMemcpy(p, h_kernels_[2 * i + 1].data(),
                                  w_ * sizeof(T) * this->projection_dim_, cudaMemcpyHostToDevice));
      } else {
        p = weights_ptrs_[ptr_idx++];
        HCTR_LIB_THROW(cudaMemcpy(p, h_kernels_[i].data(), w_ * sizeof(T), cudaMemcpyHostToDevice));
      }
      p = weights_ptrs_[ptr_idx++];
      HCTR_LIB_THROW(cudaMemcpy(p, h_biases_[i].data(), w_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    return;
  }

  void reset_backward_() {
    for (auto& a : h_output_grad_) {
      a = 1e-1;
    }
    for (auto& a : h_kernel_grads_) {
      for (auto& b : a) {
        b = 0.0f;
      }
    }
    for (auto& a : h_bias_grads_) {
      for (auto& b : a) {
        b = 0.0f;
      }
    }
    HCTR_LIB_THROW(cudaMemcpy(d_output_.data(), h_output_grad_.data(), batchsize_ * w_ * sizeof(T),
                              cudaMemcpyHostToDevice));

    int ptr_idx = 0;
    T* p;
    for (int i = 0; i < layers_; i++) {
      if (this->projection_dim_) {
        p = wgrads_ptrs_[ptr_idx++];
        HCTR_LIB_THROW(cudaMemcpy(p, h_kernel_grads_[2 * i].data(),
                                  w_ * sizeof(T) * this->projection_dim_, cudaMemcpyHostToDevice));
        p = wgrads_ptrs_[ptr_idx++];
        HCTR_LIB_THROW(cudaMemcpy(p, h_kernel_grads_[2 * i + 1].data(),
                                  w_ * sizeof(T) * this->projection_dim_, cudaMemcpyHostToDevice));
      } else {
        p = wgrads_ptrs_[ptr_idx++];
        HCTR_LIB_THROW(
            cudaMemcpy(p, h_kernel_grads_[i].data(), w_ * sizeof(T), cudaMemcpyHostToDevice));
      }
      p = wgrads_ptrs_[ptr_idx++];
      HCTR_LIB_THROW(
          cudaMemcpy(p, h_bias_grads_[i].data(), w_ * sizeof(T), cudaMemcpyHostToDevice));
    }
  }

  void matrix_vec_mul(T* out, const T* in_m, const T* in_v, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      out[j] = 0.0f;
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[j] = out[j] + T(in_m[k] * in_v[i]);
      }
    }
  }

  // gemm: C = A * B, colA = rowB
  // the rowA is always the row of op(A)
  void special_gemm(T* C, const T* A, bool transA, const T* B, bool transB, size_t rowA,
                    size_t colB, size_t rowB, float beta = 0.f) {
    // HCTR_LOG(INFO, WORLD, " inside matrix_matrix_mul rowA %ld rowB %ld\n", rowA, rowB);
    if (transB && transA) {
      // c^T
      // special_gemm(C, B, false, A, false, colB,rowA,rowB );
      HCTR_LOG(ERROR, WORLD, "not supported\n");
      return;
    }

    if (transB) {
      // HCTR_LOG(ERROR, WORLD, "trans B, rowA %d, colB %d, rowB %d\n", rowA, colB, rowB);
      for (size_t r = 0; r < rowA; r++) {
        for (size_t c = 0; c < colB; c++) {
          T acc = 0.f;
          for (size_t k = 0; k < rowB; k++) {
            // column of A is rowA
            acc = acc + (A[r * rowB + k] * B[c * rowB + k]);
          }
          C[r * colB + c] = C[r * colB + c] * beta + acc;
        }
      }
    } else if (transA) {
      // rowA == row of A^T or col of A
      for (size_t r = 0; r < rowA; r++) {
        for (size_t c = 0; c < colB; c++) {
          T acc = 0.f;
          for (size_t k = 0; k < rowB; k++) {
            // column of A is rowA
            acc = acc + (A[k * rowA + r] * B[k * colB + c]);
          }
          C[r * colB + c] = C[r * colB + c] * beta + acc;
        }
      }
    } else {
      for (size_t r = 0; r < rowA; r++) {
        for (size_t c = 0; c < colB; c++) {
          T acc = 0.f;
          for (size_t k = 0; k < rowB; k++) {
            // column of A is rowB
            acc = acc + (A[r * rowB + k] * B[k * colB + c]);
          }
          C[r * colB + c] = C[r * colB + c] * beta + acc;
        }
      }
    }
  }
  // elementwise dot:  (vec_mul)
  //       A ->  batchsizexw
  //       B ->  batchsizexw
  //       C ->  batchsizexw
  // w => input vec size
  // c => batchsize
  // TODO exchange w & batchsize placement
  void matrix_matrix_elementwise_dot(T* C, const T* A, const T* B, size_t w, size_t batchsize) {
    for (size_t r = 0; r < batchsize; r++) {
      for (size_t c = 0; c < w; c++) {
        C[r * w + c] = A[r * w + c] * B[r * w + c];
      }
    }

    /*
      for(size_t idx = 0 ;idx < w * batchsize; idx ++){
        C[idx] = A[idx] * B[idx];
      }
    */
  }
  void matrix_matrix_elementwise_dot(T* C, const T* A, const T* B, size_t w, size_t batchsize,
                                     std::ofstream& ofs) {
    for (size_t r = 0; r < batchsize; r++) {
      for (size_t c = 0; c < w; c++) {
        C[r * w + c] = A[r * w + c] * B[r * w + c];
        ofs << "C(" << r << "," << c << ") is " << A[r * w + c] << " * " << B[r * w + c] << "="
            << C[r * w + c] << std::endl;
      }
    }
  }

  void row_scaling(T* out, const T* in_m, const T* in_v, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[k] = in_m[k] * in_v[j];
      }
    }
  }

  void matrix_add(T* out, const T* in_m_1, const T* in_m_2, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[k] = in_m_1[k] + in_m_2[k];
      }
    }
  }

  void matrix_vec_add(T* out, const T* in_m, const T* in_v, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[k] = in_m[k] + in_v[i];
      }
    }
  }

  void matrix_pair_mul(T* out, const T* in_m_1, const T* in_m_2, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      out[j] = 0.0f;
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[j] = out[j] + T(in_m_1[k] * in_m_2[k]);
      }
    }
  }

  void row_scaling_sum(T* out, const T* in_m, const T* in_v, size_t h, size_t w) {
    for (size_t i = 0; i < w; i++) {
      out[i] = 0.0f;
      for (size_t j = 0; j < h; j++) {
        size_t k = j * w + i;
        out[i] = out[i] + T(in_m[k] * in_v[j]);
      }
    }
  }

  void rows_sum(T* out, const T* in_m, size_t h, size_t w) {
    for (size_t i = 0; i < w; i++) {
      out[i] = 0.0f;
      for (size_t j = 0; j < h; j++) {
        size_t k = j * w + i;
        out[i] = out[i] + in_m[k];
      }
    }
  }

  void out_product(T* out, const T* in_v_1, const T* in_v_2, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[k] = in_v_1[j] * in_v_2[i];
      }
    }
  }

  void cpu_fprop_() {
    for (int i = 0; i < layers_; i++) {
      matrix_vec_mul(h_hiddens_[i].data(), i == 0 ? h_input_.data() : h_outputs_[i - 1].data(),
                     h_kernels_[i].data(), batchsize_, w_);
      row_scaling(h_outputs_[i].data(), h_input_.data(), h_hiddens_[i].data(), batchsize_, w_);
      matrix_add(h_outputs_[i].data(), h_outputs_[i].data(),
                 i == 0 ? h_input_.data() : h_outputs_[i - 1].data(), batchsize_, w_);
      matrix_vec_add(h_outputs_[i].data(), h_outputs_[i].data(), h_biases_[i].data(), batchsize_,
                     w_);
    }
  }

  // DCN v2
  void cpu_fprop_v2_() {
    // HCTR_LOG(INFO, WORLD, "cpu_fprop_v2_ starts\n");
    // std::ofstream writer("CPUoutput_fprop.txt");
    for (int i = 0; i < layers_; i++) {
      // writer << "layer " << i << std::endl;
      // X * U
      special_gemm(XUs[i].data(), i == 0 ? h_input_.data() : h_outputs_[i - 1].data(), false,
                   h_kernels_[i * 2].data(), false, batchsize_, this->projection_dim_, w_);
      // X * U * V
      const auto& tensor_input = i == 0 ? h_input_ : h_outputs_[i - 1];
      // writer << "input"
      // z       << " is\n";
      // for (size_t b = 0; b < tensor_input.size(); b++) {
      //   if (b % w_ == 0) writer << "batch" << b / w_ << "\n";
      //   writer << "\t" << b << " " << (float)tensor_input[b] << "\n";
      // }
      // writer << "U"
      //        << " is\n";
      // for (size_t b = 0; b < h_kernels_[i * 2].size(); b++) {
      //   if (b % projection_dim_ == 0) writer << "weight" << b / projection_dim_ << "\n";
      //   writer << "\t" << b << " " << (float)h_kernels_[i * 2][b] << "\n";
      // }

      // writer << "XU"
      //        << " is\n";
      // for (size_t b = 0; b < XUs[i].size(); b++) {
      //   if (b % projection_dim_ == 0) writer << "batch" << b / projection_dim_ << "\n";
      //   writer << "\t" << b << " " << (float)XUs[i][b] << "\n";
      // }
      special_gemm(XUs[i].data(), tensor_input.data(), false, h_kernels_[i * 2].data(), false,
                   batchsize_, this->projection_dim_, w_);
      special_gemm(h_hiddens_[i].data(), XUs[i].data(), false, h_kernels_[i * 2 + 1].data(), false,
                   batchsize_, w_, this->projection_dim_);
      // HCTR_LOG(INFO,WORLD,"matrix_matrix_mul h_hiddens_[i].size() %ld OK\n",
      // h_hiddens_[i].size());
      matrix_vec_add(h_hiddens_[i].data(), h_hiddens_[i].data(), h_biases_[i].data(), batchsize_,
                     w_);
      // HCTR_LOG(INFO,WORLD,"matrix_vec_add OK\n");
      matrix_matrix_elementwise_dot(h_outputs_[i].data(), h_hiddens_[i].data(), h_input_.data(), w_,
                                    batchsize_);
      // HCTR_LOG(INFO,WORLD,"matrix_matrix_elementwise_dot OK\n");
      matrix_add(h_outputs_[i].data(), h_outputs_[i].data(),
                 i == 0 ? h_input_.data() : h_outputs_[i - 1].data(), batchsize_, w_);
    }
  }

  void gpu_fprop_() {
    layer_->fprop(true);
    HCTR_LIB_THROW(cudaDeviceSynchronize());
    return;
  }

  void cpu_bprop_() {
    std::vector<T> tmp_mat_0(batchsize_ * w_);
    std::vector<T> tmp_mat_1(batchsize_ * w_);
    std::vector<T> tmp_vec(batchsize_);
    memset(h_input_grad_.data(), 0, h_input_grad_.size() * sizeof(T));
    for (int i = layers_ - 1; i >= 0; i--) {
      row_scaling(tmp_mat_0.data(), i == layers_ - 1 ? h_output_grad_.data() : tmp_mat_1.data(),
                  h_hiddens_[i].data(), batchsize_, w_);
      matrix_add(h_input_grad_.data(), h_input_grad_.data(), tmp_mat_0.data(), batchsize_, w_);
      matrix_pair_mul(tmp_vec.data(), i == layers_ - 1 ? h_output_grad_.data() : tmp_mat_1.data(),
                      h_input_.data(), batchsize_, w_);

      // transposed_gemv
      row_scaling_sum(h_kernel_grads_[i].data(),
                      i == 0 ? h_input_.data() : h_outputs_[i - 1].data(), tmp_vec.data(),
                      batchsize_, w_);
      rows_sum(h_bias_grads_[i].data(), i == layers_ - 1 ? h_output_grad_.data() : tmp_mat_1.data(),
               batchsize_, w_);
      out_product(tmp_mat_0.data(), tmp_vec.data(), h_kernels_[i].data(), batchsize_, w_);
      matrix_add(tmp_mat_1.data(), i == layers_ - 1 ? h_output_grad_.data() : tmp_mat_1.data(),
                 tmp_mat_0.data(), batchsize_, w_);
    }
    matrix_add(h_input_grad_.data(), h_input_grad_.data(), tmp_mat_1.data(), batchsize_, w_);
  }

  void cpu_bprop_v2_() {
    std::vector<T> tmp_mat_0(batchsize_ * w_);
    std::vector<T> tmp_mat_1(batchsize_ * w_);
    std::vector<T> tmp_mat_2(batchsize_ * w_);
    std::vector<T> tmp_mat_3(batchsize_ * projection_dim_);
    memset(h_input_grad_.data(), 0, h_input_grad_.size() * sizeof(T));
    // std::ofstream writer("CPUoutput.txt");
    for (int i = layers_ - 1; i >= 0; i--) {
      {
        // writer << "layer " << i << std::endl;
        auto GRAD = i == layers_ - 1 ? h_output_grad_ : tmp_mat_1;
        // S0 = dY_i .* X , shape: (batchsize, w)
        matrix_matrix_elementwise_dot(tmp_mat_0.data(), GRAD.data(), h_input_.data(), w_,
                                      batchsize_);
        // hidden .* dY_i
        matrix_matrix_elementwise_dot(tmp_mat_2.data(), GRAD.data(), h_hiddens_[i].data(), w_,
                                      batchsize_);
        // dX , shape: (batchsize, w)
        matrix_add(h_input_grad_.data(), h_input_grad_.data(), tmp_mat_2.data(), batchsize_, w_);
        // db , shape: (1,batchsize)
        rows_sum(h_bias_grads_[i].data(), tmp_mat_0.data(), batchsize_, w_);
        // dV = XU^T * S0  , (XU is the forward pass) , shape: (projection_dim_, w)
        special_gemm(h_kernel_grads_[2 * i + 1].data(), XUs[i].data(), true, tmp_mat_0.data(),
                     false, this->projection_dim_, w_, batchsize_, 1.0);
        // S1 = S0 * V^T , shape: (batchsize, projection_dim_)
        special_gemm(tmp_mat_3.data(), tmp_mat_0.data(), false, h_kernels_[2 * i + 1].data(), true,
                     batchsize_, projection_dim_, w_);
        // dU = H^T * S1 , shape: (w, projection_dim_)
        special_gemm(h_kernel_grads_[2 * i].data(),
                     i == 0 ? h_input_.data() : h_outputs_[i - 1].data(), true, tmp_mat_3.data(),
                     false, w_, projection_dim_, batchsize_, 1.0);

        // d_hidden = S1 * U^T, shape: (batchsize, w_)
        special_gemm(tmp_mat_2.data(), tmp_mat_3.data(), false, h_kernels_[i * 2].data(), true,
                     batchsize_, w_, projection_dim_);

        // {
        // dY_{i-1} += d0
        matrix_add(tmp_mat_1.data(), i == layers_ - 1 ? h_output_grad_.data() : tmp_mat_1.data(),
                   tmp_mat_2.data(), batchsize_, w_);
        // }
      }
    }
    // accumulative dgrad of elementwise op WRT input + gemm dgrad
    matrix_add(h_input_grad_.data(), h_input_grad_.data(), tmp_mat_1.data(), batchsize_, w_);
  }

  void gpu_bprop_() {
    layer_->bprop();
    HCTR_LIB_THROW(cudaDeviceSynchronize());
    return;
  }

  void compare_forward_() {
    std::vector<T> d2h_output;
    d2h_output.resize(batchsize_ * w_);

    HCTR_LIB_THROW(cudaMemcpy(d2h_output.data(), d_output_.data(), d_output_.num_bytes(),
                              cudaMemcpyDeviceToHost));

    ASSERT_TRUE(test::compare_array_approx_rel<T>(d2h_output.data(), h_outputs_.back().data(),
                                                  h_outputs_.back().size(), 0.1f, eps_()));
  }

  void compare_hidden_() {
    std::vector<T> d2h_output;
    size_t weight_width = this->projection_dim_ ? w_ : 1;
    d2h_output.resize(batchsize_ * weight_width);
    auto& hidden_tensors = layer_->get_hidden_tensors();
    for (int i = 0; i < this->layers_; i++) {
      auto& d_tensor = hidden_tensors[i];
      HCTR_LIB_THROW(cudaMemcpy(d2h_output.data(), d_tensor.data(), d_tensor.num_bytes(),
                                cudaMemcpyDeviceToHost));
      ASSERT_TRUE(test::compare_array_approx_rel<T>(d2h_output.data(), h_hiddens_[i].data(),
                                                    h_hiddens_[i].size(), 0.1f, eps_()));
    }
  }

  void compare_XU_() {
    std::vector<T> d2h_output;
    d2h_output.resize(batchsize_ * projection_dim_);
    auto& weight_tensor = layer_->get_weight_tensor();
    for (int i = 0; i < this->layers_; i++) {
      HCTR_LIB_THROW(cudaMemcpy(d2h_output.data(), weight_tensor[i].data(),
                                weight_tensor[i].num_bytes(), cudaMemcpyDeviceToHost));
      ASSERT_TRUE(test::compare_array_approx_rel<T>(d2h_output.data(), XUs[i].data(), XUs[i].size(),
                                                    0.1f, eps_()));
    }
  }

  void compare_backward_() {
    std::vector<T> d2h_input_grad;
    std::vector<std::vector<T>> d2h_kernel_grads_;
    std::vector<std::vector<T>> d2h_bias_grads_;

    d2h_input_grad.resize(batchsize_ * w_);
    for (int i = 0; i < layers_; i++) {
      d2h_kernel_grads_.push_back(std::vector<T>(1 * w_));
      d2h_bias_grads_.push_back(std::vector<T>(1 * w_));
    }

    HCTR_LIB_THROW(cudaMemcpy(d2h_input_grad.data(), d_input_.data(), d_input_.num_bytes(),
                              cudaMemcpyDeviceToHost));

    for (int i = 0; i < layers_; i++) {
      T* p = wgrads_ptrs_[i * 2];
      HCTR_LIB_THROW(
          cudaMemcpy(d2h_kernel_grads_[i].data(), p, w_ * sizeof(T), cudaMemcpyDeviceToHost));
      p = wgrads_ptrs_[i * 2 + 1];
      HCTR_LIB_THROW(
          cudaMemcpy(d2h_bias_grads_[i].data(), p, w_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    ASSERT_TRUE(test::compare_array_approx_rel<T>(d2h_input_grad.data(), h_input_grad_.data(),
                                                  h_input_grad_.size(), 0.4f, eps_()));
    for (int i = 0; i < layers_; i++) {
      ASSERT_TRUE(test::compare_array_approx_rel<T>(d2h_kernel_grads_[i].data(),
                                                    h_kernel_grads_[i].data(),
                                                    h_kernel_grads_[i].size(), 0.4f, eps_()));
      ASSERT_TRUE(test::compare_array_approx_rel<T>(d2h_bias_grads_[i].data(),
                                                    h_bias_grads_[i].data(),
                                                    h_bias_grads_[i].size(), 0.2f, eps_()));
    }
  }

  void compare_backward_v2_() {
    std::vector<T> d2h_input_grad;
    std::vector<std::vector<T>> d2h_kernel_grads_u;
    std::vector<std::vector<T>> d2h_kernel_grads_v;
    std::vector<std::vector<T>> d2h_bias_grads_;

    d2h_input_grad.resize(batchsize_ * w_);
    for (int i = 0; i < layers_; i++) {
      d2h_kernel_grads_u.push_back(std::vector<T>(this->projection_dim_ * w_));
      d2h_kernel_grads_v.push_back(std::vector<T>(this->projection_dim_ * w_));
      d2h_bias_grads_.push_back(std::vector<T>(1 * w_));
    }

    HCTR_LIB_THROW(cudaMemcpy(d2h_input_grad.data(), d_input_.data(), d_input_.num_bytes(),
                              cudaMemcpyDeviceToHost));

    int ptr_idx = 0;
    T* p;
    for (int i = 0; i < layers_; i++) {
      p = wgrads_ptrs_[ptr_idx++];
      HCTR_LIB_THROW(cudaMemcpy(d2h_kernel_grads_u[i].data(), p,
                                d2h_kernel_grads_u[i].size() * sizeof(T), cudaMemcpyDeviceToHost));
      p = wgrads_ptrs_[ptr_idx++];
      HCTR_LIB_THROW(cudaMemcpy(d2h_kernel_grads_v[i].data(), p,
                                d2h_kernel_grads_v[i].size() * sizeof(T), cudaMemcpyDeviceToHost));
      p = wgrads_ptrs_[ptr_idx++];
      HCTR_LIB_THROW(
          cudaMemcpy(d2h_bias_grads_[i].data(), p, w_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    ASSERT_TRUE(test::compare_array_approx_rel<T>(d2h_input_grad.data(), h_input_grad_.data(),
                                                  h_input_grad_.size(), 0.4f, eps_()));
    for (int i = 0; i < layers_; i++) {
      // std::cout << " Layer " << i << std::endl;
      ASSERT_TRUE(test::compare_array_approx_rel<T>(d2h_bias_grads_[i].data(),
                                                    h_bias_grads_[i].data(),
                                                    h_bias_grads_[i].size(), 0.2f, eps_()));

      ASSERT_TRUE(test::compare_array_approx_rel<T>(
          d2h_kernel_grads_v[i].data(), h_kernel_grads_[i * 2 + 1].data(),
          h_kernel_grads_[i * 2 + 1].size(), 0.4f, eps_()));

      ASSERT_TRUE(test::compare_array_approx_rel<T>(d2h_kernel_grads_u[i].data(),
                                                    h_kernel_grads_[i * 2].data(),
                                                    h_kernel_grads_[i].size(), 0.4f, eps_()));
    }
  }

 public:
  MultiCrossLayerTest(int64_t batchsize, int64_t w, int layers, int64_t projection_dim = 0)
      : batchsize_(batchsize),
        w_(w),
        layers_(layers),
        projection_dim_(projection_dim),
        data_sim_(normal_params_()[0], normal_params_()[1]) {
    core23::BufferParams blobs_buffer_params = {};
    blobs_buffer_params.channel = GetBlobsBufferChannel();

    d_input_ = core23::Tensor(core23::TensorParams()
                                  .data_type(core23::ToScalarType<T>::value)
                                  .shape({batchsize, w})
                                  .buffer_params(blobs_buffer_params));

    d_output_ = core23::Tensor(core23::TensorParams()
                                   .data_type(core23::ToScalarType<T>::value)
                                   .shape({batchsize, w})
                                   .buffer_params(blobs_buffer_params));

    h_input_.resize(batchsize * w);
    h_output_grad_.resize(batchsize * w);
    h_input_grad_.resize(batchsize * w);
    // w = x * u
    for (int i = 0; i < layers; i++) {
      if (this->projection_dim_) {
        std::vector<T> xu(batchsize_ * this->projection_dim_);
        XUs.push_back(xu);
      }
    }

    size_t weight_width = this->projection_dim_ ? w_ : 1;
    HCTR_LOG(INFO, ROOT, "weight_width %ld\n", weight_width);
    for (int i = 0; i < layers_; i++) {
      // dcn v2
      if (this->projection_dim_) {
        // two weight tensor W=U*V
        // U
        h_kernels_.push_back(std::vector<T>(w_ * projection_dim));
        // V
        h_kernels_.push_back(std::vector<T>(projection_dim * w_));
        // dU
        h_kernel_grads_.push_back(std::vector<T>(w_ * projection_dim));
        // dV
        h_kernel_grads_.push_back(std::vector<T>(w_ * projection_dim));
      } else {
        // only one vector tensor
        h_kernels_.push_back(std::vector<T>(1 * w_));
        h_kernel_grads_.push_back(std::vector<T>(1 * w));
      }
      h_biases_.push_back(std::vector<T>(1 * w));
      h_outputs_.push_back(std::vector<T>(batchsize * w));
      h_hiddens_.push_back(std::vector<T>(batchsize * weight_width));
      h_bias_grads_.push_back(std::vector<T>(1 * w));
    }

    // layer
    layer_.reset(new Core23TempMultiCrossLayer<T>(d_input_, d_output_, test::get_default_gpu(),
                                                  layers, projection_dim_));

    layer_->initialize();

    auto weights = layer_->get_weights();
    auto weights_grad = layer_->get_wgrads();

    for (auto tensor : weights) {
      weights_ptrs_.push_back(tensor.template data<T>());
    }

    for (auto tensor : weights_grad) {
      wgrads_ptrs_.push_back(tensor.template data<T>());
    }
    return;
  }

  void test() {
    this->layer_->initialize();
    // this->layer_->search_algorithm();
    // dcnv1
    if (this->projection_dim_ == 0) {
      reset_forward_();
      cpu_fprop_();
      gpu_fprop_();
      compare_forward_();
      reset_backward_();
      cpu_bprop_();
      gpu_bprop_();
      compare_backward_();
    } else {
      // constexpr int loops = 50;
      // for(int i = 0 ; i < loops;i++){
      //   gpu_fprop_();
      // }
      // for(int i = 0 ; i < loops;i++){
      //   gpu_bprop_();
      // }
      reset_forward_();
      cpu_fprop_v2_();
      gpu_fprop_();
      compare_XU_();
      compare_hidden_();
      compare_forward_();

      reset_backward_();
      cpu_bprop_v2_();
      gpu_bprop_();
      compare_backward_v2_();
    }
  }
};

TEST(multi_cross_layer_v1, fp32_1x1024x1) {
  MultiCrossLayerTest<float> test(1, 1024, 1);
  test.test();
}

TEST(multi_cross_layer_v1, fp32_1x1024x2) {
  MultiCrossLayerTest<float> test(1, 1024, 2);
  test.test();
}

TEST(multi_cross_layer_v1, fp32_1x1024x3) {
  MultiCrossLayerTest<float> test(1, 1024, 3);
  test.test();
}

TEST(multi_cross_layer_v1, fp32_32x1024x3) {
  MultiCrossLayerTest<float> test(2, 1024, 3);
  test.test();
}

TEST(multi_cross_layer_v1, fp32_4096x1024x2) {
  MultiCrossLayerTest<float> test(4096, 1024, 2);
  test.test();
}

TEST(multi_cross_layer_v1, fp32_4096x1024x3) {
  MultiCrossLayerTest<float> test(4096, 1024, 3);
  test.test();
}

TEST(multi_cross_layer_v1, fp32_40963x356x3) {
  MultiCrossLayerTest<float> test(40963, 356, 3);
  test.test();
}
TEST(multi_cross_layer_v1, fp16_1x1024x1) {
  MultiCrossLayerTest<__half> test(1, 1024, 1);
  test.test();
}

TEST(multi_cross_layer_v1, fp16_1x1024x2) {
  MultiCrossLayerTest<__half> test(1, 1024, 2);
  test.test();
}

TEST(multi_cross_layer_v1, fp16_1x1024x3) {
  MultiCrossLayerTest<__half> test(1, 1024, 3);
  test.test();
}

TEST(multi_cross_layer_v1, fp16_32x1024x3) {
  MultiCrossLayerTest<__half> test(32, 1024, 3);
  test.test();
}

TEST(multi_cross_layer_v1, fp16_1024x1024x2) {
  MultiCrossLayerTest<__half> test(1024, 1024, 2);
  test.test();
}

TEST(multi_cross_layer_v1, fp16_1024x1024x3) {
  MultiCrossLayerTest<__half> test(1024, 1024, 3);
  test.test();
}

TEST(multi_cross_layer_v1, fp16_1283x356x3) {
  MultiCrossLayerTest<__half> test(1283, 356, 3);
  test.test();
}

// MultiCrossLayerTest(size_t batchsize, size_t w, int layers, size_t
// projection_dim = 0)
//
TEST(multi_cross_layer_v2, fp32_1x1024x1) {
  MultiCrossLayerTest<float> test(1, 256, 1, 256);
  MultiCrossLayerTest<float> test1(1, 1024, 1, 256);
  MultiCrossLayerTest<float> test2(4, 256, 1, 256);
  MultiCrossLayerTest<float> test3(4, 1024, 1, 256);
  MultiCrossLayerTest<float> test4(8, 1024, 1, 128);
  test.test();
  test1.test();
  test2.test();
  test3.test();
  test4.test();
}
TEST(multi_cross_layer_v2, fp32_4096x356x3) {
  MultiCrossLayerTest<float> test(4096, 256, 3, 512);
  test.test();
}
// MultiCrossLayerTest(size_t batchsize, size_t w, int layers, size_t projection_dim = 0)
TEST(multi_cross_layer_v2, fp32_1x256x1) {
  MultiCrossLayerTest<float> test(2, 1024, 4, 256);
  test.test();
}
TEST(multi_cross_layer_v2, fp32_1x1024x2) {
  MultiCrossLayerTest<float> test(1, 1024, 2, 256);
  test.test();
}
// TEST(multi_cross_layer_v2, fp32_3x1024x2) {
//   MultiCrossLayerTest<float> test(3, 1024, 2, 1024);
//   test.test();
// }
TEST(multi_cross_layer_v2, fp32_3x1024x3) {
  MultiCrossLayerTest<float> test(3, 1024, 3, 256);
  test.test();
}
TEST(multi_cross_layer_v2, fp32_3x1024x10) {
  MultiCrossLayerTest<float> test(3, 1024, 10, 64);
  test.test();
}
TEST(multi_cross_layer_v2, fp32_3x1024x2) {
  MultiCrossLayerTest<float> test(3, 1024, 2, 256);
  test.test();
}
TEST(multi_cross_layer_v2, fp32_3x1024x4) {
  MultiCrossLayerTest<float> test(3, 1024, 4, 256);
  test.test();
}
//
TEST(multi_cross_layer_v2, fp16_debug) {
  MultiCrossLayerTest<__half> test0(1, 1024, 1, 1024);
  MultiCrossLayerTest<__half> test1(1, 1024, 1, 512);
  MultiCrossLayerTest<__half> test2(1, 256, 3, 256);
  MultiCrossLayerTest<__half> test3(3, 512, 3, 1024);
  MultiCrossLayerTest<__half> test4(3, 1024, 3, 32);
  MultiCrossLayerTest<__half> test5(3, 1024, 3, 64);
  MultiCrossLayerTest<__half> test6(3, 1024, 3, 128);
  MultiCrossLayerTest<__half> test7(3, 1024, 3, 256);
  MultiCrossLayerTest<__half> test8(3, 1024, 3, 512);
  MultiCrossLayerTest<__half> test9(3, 3456, 3, 256);
  MultiCrossLayerTest<__half> test10(3, 3456, 3, 64);
  MultiCrossLayerTest<__half> test11(3, 3456, 3, 512);
  MultiCrossLayerTest<__half> test12(3, 3456, 3, 512);
  MultiCrossLayerTest<__half> test13(1024, 1024, 1, 1024);
  test0.test();
  std::cout << "Test0 \n";
  test1.test();
  std::cout << "Test1 \n";
  test2.test();
  std::cout << "Test2 \n";
  test3.test();
  std::cout << "Test3 \n";
  test4.test();
  std::cout << "Test4 \n";
  test5.test();
  std::cout << "Test5 \n";
  test6.test();
  std::cout << "Test6 \n";
  test7.test();
  std::cout << "Test7 \n";
  test8.test();
  std::cout << "Test8 \n";
  test9.test();
  std::cout << "Test9 \n";
  test10.test();
  std::cout << "Test10 \n";
  test11.test();
  std::cout << "Test11 \n";
  test12.test();
  std::cout << "Test12 \n";
  test13.test();
  std::cout << "Test13 \n";
}

TEST(multi_cross_layer_v2, fp16_3x1024x1) {
  MultiCrossLayerTest<__half> test(3, 1024, 1, 1024);
  test.test();
}
TEST(multi_cross_layer_v2, fp16_1x1024x2) {
  MultiCrossLayerTest<__half> test(1, 1024, 2, 1024);
  test.test();
}
TEST(multi_cross_layer_v2, fp16_3x1024x2) {
  MultiCrossLayerTest<__half> test(3, 1024, 2, 1024);
  test.test();
}

// TEST(multi_cross_layer_v2, fp16_dlrm) {
//   MultiCrossLayerTest<__half> test(65536, 3456, 3, 512);
//   test.test();
// }
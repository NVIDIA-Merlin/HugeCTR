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

#include "HugeCTR/include/layers/multi_cross_layer.hpp"

#include <math.h>

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

class MultiCrossLayerTest {
 private:
  const float eps = 1;
  const size_t batchsize_;
  const size_t w_;
  const int layers_;
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> blob_buf_;
  std::shared_ptr<BufferBlock2<float>> weight_buf_;
  std::shared_ptr<BufferBlock2<float>> wgrad_buf_;

  Tensor2<float> weight_;
  Tensor2<float> wgrad_;

  Tensor2<float> d_input_;
  Tensor2<float> d_output_;

  std::vector<float> h_input_;
  std::vector<float> h_input_grad_;
  std::vector<float> h_output_grad_;
  std::vector<std::vector<float>> h_kernels_;
  std::vector<std::vector<float>> h_biases_;

  std::vector<std::vector<float>> h_outputs_;
  std::vector<std::vector<float>> h_hiddens_;

  std::vector<std::vector<float>> h_kernel_grads_;
  std::vector<std::vector<float>> h_bias_grads_;

  std::shared_ptr<MultiCrossLayer> layer_;
  test::GaussianDataSimulator data_sim_;

  void reset_forward_() {
    data_sim_.fill(h_input_.data(), batchsize_ * w_);
    for (auto& a : h_kernels_) {
      data_sim_.fill(a.data(), w_);
    }
    for (auto& a : h_biases_) {
      data_sim_.fill(a.data(), w_);
    }

    CK_CUDA_THROW_(cudaMemcpy(d_input_.get_ptr(), h_input_.data(), d_input_.get_size_in_bytes(),
                              cudaMemcpyHostToDevice));
    float* p = weight_.get_ptr();
    for (int i = 0; i < layers_; i++) {
      CK_CUDA_THROW_(
          cudaMemcpy(p, h_kernels_[i].data(), w_ * sizeof(float), cudaMemcpyHostToDevice));
      p += w_;
      CK_CUDA_THROW_(
          cudaMemcpy(p, h_biases_[i].data(), w_ * sizeof(float), cudaMemcpyHostToDevice));
      p += w_;
    }
    return;
  }

  void reset_backward_() {
    for (auto& a : h_output_grad_) {
      a = 1.0f;
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
    CK_CUDA_THROW_(cudaMemcpy(d_output_.get_ptr(), h_output_grad_.data(),
                              batchsize_ * w_ * sizeof(float), cudaMemcpyHostToDevice));
    float* p = wgrad_.get_ptr();
    for (int i = 0; i < layers_; i++) {
      CK_CUDA_THROW_(
          cudaMemcpy(p, h_kernel_grads_[i].data(), w_ * sizeof(float), cudaMemcpyHostToDevice));
      p += w_;
      CK_CUDA_THROW_(
          cudaMemcpy(p, h_bias_grads_[i].data(), w_ * sizeof(float), cudaMemcpyHostToDevice));
      p += w_;
    }
  }

  void matrix_vec_mul(float* out, const float* in_m, const float* in_v, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      out[j] = 0.0f;
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[j] += in_m[k] * in_v[i];
      }
    }
  }

  void row_scaling(float* out, const float* in_m, const float* in_v, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[k] = in_m[k] * in_v[j];
      }
    }
  }

  void matrix_add(float* out, const float* in_m_1, const float* in_m_2, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[k] = in_m_1[k] + in_m_2[k];
      }
    }
  }

  void matrix_vec_add(float* out, const float* in_m, const float* in_v, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[k] = in_m[k] + in_v[i];
      }
    }
  }

  void matrix_pair_mul(float* out, const float* in_m_1, const float* in_m_2, size_t h, size_t w) {
    for (size_t j = 0; j < h; j++) {
      out[j] = 0.0f;
      for (size_t i = 0; i < w; i++) {
        size_t k = j * w + i;
        out[j] += in_m_1[k] * in_m_2[k];
      }
    }
  }

  void row_scaling_sum(float* out, const float* in_m, const float* in_v, size_t h, size_t w) {
    for (size_t i = 0; i < w; i++) {
      out[i] = 0.0f;
      for (size_t j = 0; j < h; j++) {
        size_t k = j * w + i;
        out[i] += in_m[k] * in_v[j];
      }
    }
  }

  void rows_sum(float* out, const float* in_m, size_t h, size_t w) {
    for (size_t i = 0; i < w; i++) {
      out[i] = 0.0f;
      for (size_t j = 0; j < h; j++) {
        size_t k = j * w + i;
        out[i] += in_m[k];
      }
    }
  }

  void out_product(float* out, const float* in_v_1, const float* in_v_2, size_t h, size_t w) {
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

  void gpu_fprop_() {
    layer_->fprop(true);
    CK_CUDA_THROW_(cudaDeviceSynchronize());
    return;
  }

  void cpu_bprop_() {
    std::vector<float> tmp_mat_0(batchsize_ * w_);
    std::vector<float> tmp_mat_1(batchsize_ * w_);
    std::vector<float> tmp_vec(batchsize_);
    memset(h_input_grad_.data(), 0, h_input_grad_.size() * sizeof(float));
    for (int i = layers_ - 1; i >= 0; i--) {
      row_scaling(tmp_mat_0.data(), i == layers_ - 1 ? h_output_grad_.data() : tmp_mat_1.data(),
                  h_hiddens_[i].data(), batchsize_, w_);
      matrix_add(h_input_grad_.data(), h_input_grad_.data(), tmp_mat_0.data(), batchsize_, w_);
      matrix_pair_mul(tmp_vec.data(), i == layers_ - 1 ? h_output_grad_.data() : tmp_mat_1.data(),
                      h_input_.data(), batchsize_, w_);
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

  void gpu_bprop_() {
    layer_->bprop();
    CK_CUDA_THROW_(cudaDeviceSynchronize());
    return;
  }

  void compare_forward_() {
    std::vector<float> d2h_output;
    d2h_output.resize(batchsize_ * w_);

    CK_CUDA_THROW_(cudaMemcpy(d2h_output.data(), d_output_.get_ptr(), d_output_.get_size_in_bytes(),
                              cudaMemcpyDeviceToHost));

    // todo compare
    ASSERT_TRUE(test::compare_array_approx_with_ratio<float>(
        d2h_output.data(), h_outputs_.back().data(), h_outputs_.back().size(), eps, 0.05f));
  }

  void compare_backward_() {
    std::vector<float> d2h_input_grad;
    std::vector<std::vector<float>> d2h_kernel_grads_;
    std::vector<std::vector<float>> d2h_bias_grads_;

    d2h_input_grad.resize(batchsize_ * w_);
    for (int i = 0; i < layers_; i++) {
      d2h_kernel_grads_.push_back(std::vector<float>(1 * w_));
      d2h_bias_grads_.push_back(std::vector<float>(1 * w_));
    }

    CK_CUDA_THROW_(cudaMemcpy(d2h_input_grad.data(), d_input_.get_ptr(),
                              d_input_.get_size_in_bytes(), cudaMemcpyDeviceToHost));
    float* p = wgrad_.get_ptr();
    for (int i = 0; i < layers_; i++) {
      CK_CUDA_THROW_(
          cudaMemcpy(d2h_kernel_grads_[i].data(), p, w_ * sizeof(float), cudaMemcpyDeviceToHost));
      p += w_;
      CK_CUDA_THROW_(
          cudaMemcpy(d2h_bias_grads_[i].data(), p, w_ * sizeof(float), cudaMemcpyDeviceToHost));
      p += w_;
    }

    ASSERT_TRUE(test::compare_array_approx_with_ratio<float>(
        d2h_input_grad.data(), h_input_grad_.data(), h_input_grad_.size(), eps, 0.05f));
    for (int i = 0; i < layers_; i++) {
      ASSERT_TRUE(test::compare_array_approx_with_ratio<float>(
          d2h_kernel_grads_[i].data(), h_kernel_grads_[i].data(), h_kernel_grads_[i].size(),
          eps * 10, 0.40f));
      ASSERT_TRUE(test::compare_array_approx_with_ratio<float>(
          d2h_bias_grads_[i].data(), h_bias_grads_[i].data(), h_bias_grads_[i].size(), eps * 10,
          0.20f));
    }
  }

 public:
  MultiCrossLayerTest(size_t batchsize, size_t w, int layers)
      : batchsize_(batchsize),
        w_(w),
        layers_(layers),
        blob_buf_(GeneralBuffer2<CudaAllocator>::create()),
        data_sim_(0.0f, 1.0f) {
    weight_buf_ = blob_buf_->create_block<float>();
    wgrad_buf_ = blob_buf_->create_block<float>();

    blob_buf_->reserve({batchsize, w}, &d_input_);
    blob_buf_->reserve({batchsize, w}, &d_output_);

    h_input_.resize(batchsize * w);
    h_output_grad_.resize(batchsize * w);
    h_input_grad_.resize(batchsize * w);

    for (int i = 0; i < layers_; i++) {
      h_kernels_.push_back(std::vector<float>(1 * w));
      h_biases_.push_back(std::vector<float>(1 * w));
      h_outputs_.push_back(std::vector<float>(batchsize * w));
      h_hiddens_.push_back(std::vector<float>(batchsize * 1));
      h_kernel_grads_.push_back(std::vector<float>(1 * w));
      h_bias_grads_.push_back(std::vector<float>(1 * w));
    }

    // layer
    layer_.reset(new MultiCrossLayer(weight_buf_, wgrad_buf_, blob_buf_, d_input_, d_output_,
                                     test::get_default_gpu(), layers));

    blob_buf_->allocate();
    layer_->initialize();

    weight_ = weight_buf_->as_tensor();
    wgrad_ = wgrad_buf_->as_tensor();

    return;
  }

  void test() {
    reset_forward_();
    cpu_fprop_();
    gpu_fprop_();
    compare_forward_();
    reset_backward_();
    cpu_bprop_();
    gpu_bprop_();
    compare_backward_();
  }
};

TEST(multi_cross_layer, fp32_1x1024x1) {
  MultiCrossLayerTest test(1, 1024, 1);
  test.test();
}

TEST(multi_cross_layer, fp32_1x1024x2) {
  MultiCrossLayerTest test(1, 1024, 2);
  test.test();
}

TEST(multi_cross_layer, fp32_1x1024x3) {
  MultiCrossLayerTest test(1, 1024, 3);
  test.test();
}

TEST(multi_cross_layer, fp32_32x1024x3) {
  MultiCrossLayerTest test(32, 1024, 3);
  test.test();
}

TEST(multi_cross_layer, fp32_4096x1024x2) {
  MultiCrossLayerTest test(4096, 1024, 2);
  test.test();
}

TEST(multi_cross_layer, fp32_4096x1024x3) {
  MultiCrossLayerTest test(4096, 1024, 3);
  test.test();
}

TEST(multi_cross_layer, fp32_40963x356x3) {
  MultiCrossLayerTest test(40963, 356, 3);
  test.test();
}

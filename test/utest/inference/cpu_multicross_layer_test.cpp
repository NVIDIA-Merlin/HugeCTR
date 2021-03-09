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

#include "HugeCTR/include/cpu/layers/multi_cross_layer_cpu.hpp"
#include "HugeCTR/include/layer.hpp"
#include <math.h>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

class MultiCrossLayerCPUTest {
 private:
  const float eps = 1;
  const size_t batchsize_;
  const size_t w_;
  const int layers_;
  std::shared_ptr<GeneralBuffer2<HostAllocator>> blob_buf_;
  std::shared_ptr<BufferBlock2<float>> weight_buf_;
  std::shared_ptr<BufferBlock2<float>> wgrad_buf_;

  Tensor2<float> weight_;
  Tensor2<float> wgrad_;

  Tensor2<float> input_;
  Tensor2<float> output_;

  std::vector<float> h_input_;
  std::vector<float> h_input_grad_;
  std::vector<float> h_output_grad_;
  std::vector<std::vector<float>> h_kernels_;
  std::vector<std::vector<float>> h_biases_;

  std::vector<std::vector<float>> h_outputs_;
  std::vector<std::vector<float>> h_hiddens_;

  std::vector<std::vector<float>> h_kernel_grads_;
  std::vector<std::vector<float>> h_bias_grads_;

  std::shared_ptr<MultiCrossLayerCPU> layer_;
  test::GaussianDataSimulator data_sim_;

  void reset_forward_() {
    data_sim_.fill(h_input_.data(), batchsize_ * w_);
    for (auto& a : h_kernels_) {
      data_sim_.fill(a.data(), w_);
    }
    for (auto& a : h_biases_) {
      data_sim_.fill(a.data(), w_);
    }
    memcpy(input_.get_ptr(), h_input_.data(), input_.get_size_in_bytes());

    float* p = weight_.get_ptr();
    for (int i = 0; i < layers_; i++) {
      memcpy(p, h_kernels_[i].data(), w_ * sizeof(float));
      p += w_;
      memcpy(p, h_biases_[i].data(), w_ * sizeof(float));
      p += w_;
    }
    return;
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

  void layer_fprop_() {
    layer_->fprop(false);
    return;
  }

  void compare_forward_() {
    std::vector<float> d2h_output;
    d2h_output.resize(batchsize_ * w_);

    memcpy(d2h_output.data(), output_.get_ptr(), output_.get_size_in_bytes());

    // todo compare
    for (size_t i = 0; i < h_outputs_.back().size(); i++) {
      if (abs(d2h_output[i]-h_outputs_.back()[i]) > 0.05f) {
        CK_THROW_(Error_t::WrongInput, "cpu multicross layer wrong result");
      }
    }
  }

 public:
  MultiCrossLayerCPUTest(size_t batchsize, size_t w, int layers)
      : batchsize_(batchsize),
        w_(w),
        layers_(layers),
        blob_buf_(GeneralBuffer2<HostAllocator>::create()),
        data_sim_(0.0f, 1.0f) {
    weight_buf_ = blob_buf_->create_block<float>();
    wgrad_buf_ = blob_buf_->create_block<float>();

    blob_buf_->reserve({batchsize, w}, &input_);
    blob_buf_->reserve({batchsize, w}, &output_);

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
    layer_.reset(new MultiCrossLayerCPU(weight_buf_, wgrad_buf_, blob_buf_, input_, output_, layers));

    blob_buf_->allocate();
    layer_->initialize();

    weight_ = weight_buf_->as_tensor();
    wgrad_ = wgrad_buf_->as_tensor();

    return;
  }

  void test() {
    reset_forward_();
    cpu_fprop_();
    layer_fprop_();
    compare_forward_();
  }
};

TEST(multi_cross_layer_cpu, fp32_1x4x1) {
  MultiCrossLayerCPUTest test(1, 4, 1);
  test.test();
}

TEST(multi_cross_layer_cpu, fp32_1x1024x2) {
  MultiCrossLayerCPUTest test(1, 1024, 2);
  test.test();
}

TEST(multi_cross_layer_cpu, fp32_1x1024x3) {
  MultiCrossLayerCPUTest test(1, 1024, 3);
  test.test();
}

TEST(multi_cross_layer_cpu, fp32_32x1024x3) {
  MultiCrossLayerCPUTest test(32, 1024, 3);
  test.test();
}

// TEST(multi_cross_layer_cpu, fp32_4096x1024x2) {
//   MultiCrossLayerCPUTest test(4096, 1024, 2);
//   test.test();
// }

// TEST(multi_cross_layer_cpu, fp32_4096x1024x3) {
//   MultiCrossLayerCPUTest test(4096, 1024, 3);
//   test.test();
// }

// TEST(multi_cross_layer_cpu, fp32_40963x356x3) {
//   MultiCrossLayerCPUTest test(40963, 356, 3);
//   test.test();
// }

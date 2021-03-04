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

#include <math.h>
#include <utils.hpp>
#include <vector>
#include "tools/cpu_inference/layers/include/multi_cross_layer_cpu.hpp"

namespace HugeCTR {

namespace {

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

void multi_cross_fprop_cpu(int layers, size_t batchsize, size_t w, float** h_outputs,
                        float* h_input, float** h_hiddens, float** h_kernels, float** h_biases) {
  for (int i = 0; i < layers; i++) {
    matrix_vec_mul(h_hiddens[i], i == 0 ? h_input : h_outputs[i - 1],
                    h_kernels[i], batchsize, w);
    row_scaling(h_outputs[i], h_input, h_hiddens[i], batchsize, w);
    matrix_add(h_outputs[i], h_outputs[i],
                i == 0 ? h_input : h_outputs[i - 1], batchsize, w);
    matrix_vec_add(h_outputs[i], h_outputs[i], h_biases[i], batchsize, w);
  }
}

}  // namespace


MultiCrossLayerCPU::MultiCrossLayerCPU(const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                                 const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                                 const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff,
                                 const Tensor2<float>& in_tensor, const Tensor2<float>& out_tensor,
                                 int num_layers)
    : LayerCPU(), num_layers_(num_layers) {
  try {
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor.get_dimensions();
    const auto& out_tensor_dim = out_tensor.get_dimensions();
    // 1. two dim?
    if (in_tensor_dim.size() != 2 || out_tensor_dim.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
    }
    // 2. same dim?
    for (int i = 0; i < 2; i++) {
      if (in_tensor_dim[i] != out_tensor_dim[i]) {
        CK_THROW_(Error_t::WrongInput, "input and output tensor doesn't match");
      }
    }
    size_t vec_length = in_tensor_dim[1];
    size_t batchsize = in_tensor_dim[0];

    // check num_lyaers
    if (num_layers < 1) {
      CK_THROW_(Error_t::WrongInput, "num_layers < 1");
    }

    std::vector<size_t> weight_bias_dim = {1, vec_length};
    for (int i = 0; i < num_layers; i++) {
      // setup weights
      {
        Tensor2<float> tensor;
        weight_buff->reserve(weight_bias_dim, &tensor);
        weights_.push_back(tensor);
      }
      // setup bias
      {
        Tensor2<float> tensor;
        weight_buff->reserve(weight_bias_dim, &tensor);
        weights_.push_back(tensor);
      }
      // setup weight gradient
      {
        Tensor2<float> tensor;
        wgrad_buff->reserve(weight_bias_dim, &tensor);
        wgrad_.push_back(tensor);
      }
      // setup bias gradient
      {
        Tensor2<float> tensor;
        wgrad_buff->reserve(weight_bias_dim, &tensor);
        wgrad_.push_back(tensor);
      }
    }

    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);
    // setup blobs

    std::vector<size_t> blob_dim = {batchsize, vec_length};
    blob_tensors_.push_back(in_tensor);
    for (int i = 0; i < num_layers - 1; i++) {
      Tensor2<float> tensor;
      blobs_buff->reserve(blob_dim, &tensor);
      blob_tensors_.push_back(tensor);
    }
    blob_tensors_.push_back(out_tensor);

    for (int i = 0; i < 3; i++) {
      blobs_buff->reserve(blob_dim, &tmp_mat_tensors_[i]);
    }
    std::vector<size_t> tmp_vec_dim = {batchsize, 1};
    blobs_buff->reserve(tmp_vec_dim, &tmp_vec_tensor_);
    for (int i = 0; i < num_layers; i++) {
      Tensor2<float> tensor;
      blobs_buff->reserve(tmp_vec_dim, &tensor);
      vec_tensors_.push_back(tensor);
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void MultiCrossLayerCPU::fprop(bool is_train) {
  size_t vec_length = in_tensors_[0].get_dimensions()[1];
  size_t batchsize = in_tensors_[0].get_dimensions()[0];
  Tensors2<float> kernel_tensors;
  Tensors2<float> bias_tensors;
  Tensors2<float> output_tensors;
  Tensors2<float> hidden_tensors;

  for (int i = 0; i < num_layers_; i++) {
    kernel_tensors.push_back(weights_[2 * i]);
    bias_tensors.push_back(weights_[2 * i + 1]);
  }

  for (int i = 0; i < num_layers_; i++) {
    output_tensors.push_back(blob_tensors_[i + 1]);
    hidden_tensors.push_back(vec_tensors_[i]);
  }
  std::vector<float*> h_hiddens;
  std::vector<float*> h_kernels;
  std::vector<float*> h_biases;
  std::vector<float*> h_outputs;
  for (int i = 0; i < num_layers_; i++) {
    h_hiddens.push_back(hidden_tensors[i].get_ptr());
    h_kernels.push_back(kernel_tensors[i].get_ptr());
    h_biases.push_back(bias_tensors[i].get_ptr());
    h_outputs.push_back(output_tensors[i].get_ptr());
  }
  multi_cross_fprop_cpu(num_layers_, batchsize, vec_length, h_outputs.data(),
                      blob_tensors_[0].get_ptr(), h_hiddens.data(), h_kernels.data(), h_biases.data());
}

void MultiCrossLayerCPU::bprop() {}

}  // namespace HugeCTR

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

#include <gtest/gtest.h>

#include <general_buffer2.hpp>
#include <layers/layer_norm_layer.hpp>
#include <utest/test_utils.hpp>

using namespace std;
using namespace HugeCTR;

namespace {

constexpr float eps = 1e-4;  // Epsilon for CPU computation

// Eps type for error
template <typename T>
struct Eps {
  static T value();
};

template <>
struct Eps<float> {
  static constexpr float value() { return 1e-3f; }
};

template <>
struct Eps<__half> {
  static __half value() { return __float2half(1e-2f); }
};

template <typename T>
void layer_norm_fprop_cpu(const T* gamma, const T* beta, const T* in, T* out, int batch_size,
                          int num_feature) {
  for (int i = 0; i < batch_size; i++) {
    float mean = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      mean += in[idx];
    }
    mean /= num_feature;

    float var = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      float diff = in[idx] - mean;
      var += (diff * diff);
    }
    var /= num_feature;

    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      float in_norm = (in[idx] - mean) / sqrt(var + eps);
      out[idx] = gamma[j] * in_norm + beta[j];
    }
  }
}

template <>
void layer_norm_fprop_cpu<__half>(const __half* gamma, const __half* beta, const __half* in,
                                  __half* out, int batch_size, int num_feature) {
  for (int i = 0; i < batch_size; i++) {
    float mean = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      mean += __half2float(in[idx]);
    }
    mean /= num_feature;

    float var = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      float diff = __half2float(in[idx]) - mean;
      var += (diff * diff);
    }
    var /= num_feature;

    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      float in_norm = (__half2float(in[idx]) - mean) / sqrt(var + eps);
      out[idx] = __float2half(gamma[j] * in_norm + beta[j]);
    }
  }
}

template <typename T>
void layer_norm_bprop_cpu(const T* gamma, const T* out, T* in, T* gamma_grad, T* beta_grad,
                          int batch_size, int num_feature) {
  memset(gamma_grad, 0.0f, num_feature * sizeof(T));
  memset(beta_grad, 0.0f, num_feature * sizeof(T));
  for (int i = 0; i < batch_size; i++) {
    float mean = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      mean += in[idx];
    }
    mean /= num_feature;

    float var = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      float diff = in[idx] - mean;
      var += (diff * diff);
    }
    var /= num_feature;

    float inv_std = 1.0f / sqrt(var + eps);

    float d_var = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      float val = (out[idx] * gamma[j]) * (in[idx] - mean);
      d_var += val;
    }

    d_var *= (-0.5f) * pow(inv_std, 3);

    float d_mu = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      d_mu += (out[idx] * gamma[j] * inv_std);
    }
    d_mu *= (-1.0f / num_feature);

    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      gamma_grad[j] += out[idx] * (in[idx] - mean) * inv_std;
      beta_grad[j] += out[idx];

      in[idx] =
          (out[idx] * gamma[j]) * inv_std + d_var * (2.0 * (in[idx] - mean) / num_feature) + d_mu;
    }
  }
}

template <>
void layer_norm_bprop_cpu<__half>(const __half* gamma, const __half* out, __half* in,
                                  __half* gamma_grad, __half* beta_grad, int batch_size,
                                  int num_feature) {
  for (int j = 0; j < num_feature; j++) {
    gamma_grad[j] = __float2half(0.0f);
    beta_grad[j] = __float2half(0.0f);
  }
  for (int i = 0; i < batch_size; i++) {
    float mean = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      mean += __half2float(in[idx]);
    }
    mean /= num_feature;

    float var = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      float diff = __half2float(in[idx]) - mean;
      var += (diff * diff);
    }
    var /= num_feature;

    float inv_std = 1.0f / sqrt(var + eps);

    float d_var = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      float val = (__half2float(out[idx]) * gamma[j]) * (__half2float(in[idx]) - mean);
      d_var += val;
    }
    d_var *= (-0.5f) * pow(inv_std, 3);

    float d_mu = 0.0f;
    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      d_mu += __half2float(out[idx]) * gamma[j] * inv_std;
    }
    d_mu *= (-1.0f / num_feature);

    for (int j = 0; j < num_feature; j++) {
      int idx = i * num_feature + j;
      gamma_grad[j] = gamma_grad[j] + out[idx] * (in[idx] - mean) * inv_std;
      beta_grad[j] = beta_grad[j] + out[idx];
      in[idx] = __float2half((__half2float(out[idx]) * gamma[j]) * inv_std +
                             d_var * (2.0 / num_feature) * (__half2float(in[idx]) - mean) + d_mu);
    }
  }
}

template <typename T>
void layer_norm_test(std::vector<size_t> dims) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  std::shared_ptr<BufferBlock2<T>> wbuff = buff->create_block<T>();
  std::shared_ptr<BufferBlock2<float>> master_weight_buff = buff->create_block<float>();
  std::shared_ptr<BufferBlock2<T>> wgbuff = buff->create_block<T>();

  Tensor2<T> in_tensor;
  buff->reserve(dims, &in_tensor);
  Tensor2<T> out_tensor;
  buff->reserve(dims, &out_tensor);

  typename LayerNormLayer<T>::Params params = {eps};
  LayerNormLayer<T> layer_norm_layer(master_weight_buff, wbuff, wgbuff, buff, in_tensor, out_tensor,
                                     params, test::get_default_gpu());

  buff->allocate();

  const auto& in_tensor_dim = dims;
  size_t batch_size = 1;
  size_t num_feature = in_tensor_dim[in_tensor_dim.size() - 1];

  for (size_t idx = 0; idx < in_tensor_dim.size(); idx++) {
    cout << in_tensor_dim[idx] << endl;
  }

  for (size_t idx = 0; idx < in_tensor_dim.size() - 1; idx++) {
    batch_size = batch_size * in_tensor_dim[idx];
  }
  const size_t len = batch_size * num_feature;

  cout << "Batch_size: " << batch_size << " Num_feature: " << num_feature << endl;

  T* d_in = in_tensor.get_ptr();
  T* d_out = out_tensor.get_ptr();

  std::unique_ptr<T[]> h_gamma(new T[num_feature]);
  std::unique_ptr<T[]> h_beta(new T[num_feature]);
  std::unique_ptr<T[]> h_gamma_grad(new T[num_feature]);
  std::unique_ptr<T[]> h_beta_grad(new T[num_feature]);
  std::unique_ptr<T[]> h_gamma_grad_expected(new T[num_feature]);
  std::unique_ptr<T[]> h_beta_grad_expected(new T[num_feature]);

  std::unique_ptr<T[]> h_in(new T[len]);
  std::unique_ptr<T[]> h_out(new T[len]);
  std::unique_ptr<T[]> h_expected(new T[len]);

  test::GaussianDataSimulator simulator(0.0, 1.0);

  // standard normall distribution is assumed
  for (size_t j = 0; j < num_feature; j++) {
    h_gamma[j] = 1.0f;
    h_beta[j] = 0.0f;
  }

  Tensor2<T> weight_tensor = wbuff->as_tensor();
  Tensor2<T> weight_grad_tensor = wgbuff->as_tensor();

  T* d_gamma = weight_tensor.get_ptr();
  T* d_beta = weight_tensor.get_ptr() + num_feature;
  T* d_gamma_grad = weight_grad_tensor.get_ptr();
  T* d_beta_grad = weight_grad_tensor.get_ptr() + num_feature;
  HCTR_LIB_THROW(
      cudaMemcpy(d_gamma, h_gamma.get(), num_feature * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(d_beta, h_beta.get(), num_feature * sizeof(T), cudaMemcpyHostToDevice));

  simulator.fill(h_in.get(), len);

  layer_norm_fprop_cpu<T>(h_gamma.get(), h_beta.get(), h_in.get(), h_expected.get(), batch_size,
                          num_feature);

  HCTR_LIB_THROW(cudaMemcpy(d_in, h_in.get(), len * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  layer_norm_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(cudaMemcpy(h_out.get(), d_out, len * sizeof(T), cudaMemcpyDeviceToHost));

  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_expected.get(), len, Eps<T>::value()));

  simulator.fill(h_out.get(), len);

  HCTR_LIB_THROW(cudaMemcpy(h_expected.get(), d_in, len * sizeof(T), cudaMemcpyDeviceToHost));

  layer_norm_bprop_cpu<T>(h_gamma.get(), h_out.get(), h_expected.get(), h_gamma_grad.get(),
                          h_beta_grad.get(), batch_size, num_feature);

  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), len * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  layer_norm_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::shared_ptr<GPUResource> gpu_resource = test::get_default_gpu();
  cudaStreamSynchronize(gpu_resource->get_stream());
  cout << cudaGetLastError() << endl;

  HCTR_LIB_THROW(cudaMemcpy(h_in.get(), d_in, len * sizeof(T), cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(cudaMemcpy(h_gamma_grad_expected.get(), d_gamma_grad, num_feature * sizeof(T),
                            cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(cudaMemcpy(h_beta_grad_expected.get(), d_beta_grad, num_feature * sizeof(T),
                            cudaMemcpyDeviceToHost));

  ASSERT_TRUE(test::compare_array_approx<T>(h_in.get(), h_expected.get(), len, Eps<T>::value()));
  ASSERT_TRUE(test::compare_array_approx<T>(h_gamma_grad.get(), h_gamma_grad_expected.get(),
                                            num_feature, Eps<T>::value()));
  ASSERT_TRUE(test::compare_array_approx<T>(h_beta_grad.get(), h_beta_grad_expected.get(),
                                            num_feature, Eps<T>::value()));
}

}  // namespace

TEST(layer_norm_layer, fp32_2x512) {
  std::vector<size_t> dims = {2, 512};
  layer_norm_test<float>(dims);
}
TEST(layer_norm_layer, fp32_4x2048) {
  std::vector<size_t> dims = {4, 2048};
  layer_norm_test<float>(dims);
}
TEST(layer_norm_layer, fp32_4x10x1024) {
  std::vector<size_t> dims{4, 10, 1024};
  layer_norm_test<float>(dims);
}
TEST(layer_norm_layer, fp32_1x1024x2x768) {
  std::vector<size_t> dims{1, 1024, 2, 768};
  layer_norm_test<float>(dims);
}
TEST(layer_norm_layer, fp16_2x1024) {
  std::vector<size_t> dims{2, 1024};
  layer_norm_test<__half>(dims);
}
/*TEST(layer_norm_layer, fp16_2x1024x20x512) {
  std::vector<size_t> dims{1, 4, 1, 768};
  layer_norm_test<__half>(dims);
}*/

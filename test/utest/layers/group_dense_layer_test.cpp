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

#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <gtest/gtest.h>
#include <nvToolsExt.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <layer.hpp>
#include <layers/fused_relu_bias_fully_connected_layer.hpp>
#include <layers/interaction_layer.hpp>
#include <string>
#include <utest/test_utils.hpp>
#include <vector>

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

using namespace HugeCTR;

namespace {
struct ConfigSet {
  bool is_perf_test;
  bool use_nvtx;
  bool use_record;
  bool use_cuda_graph;
  bool async_mlp_wgrad;
  size_t test_loop_cnt;
  size_t layer_loop_cnt;
};

static void fill_data(__half* data, int N) {
  unsigned seed = time(0);
  srand(seed);
  for (int i = 0; i < N; i++) {
    data[i] = (__half((float)(rand() % 3 - 1)));
    if (rand() % 50) {
      data[i] = (__half((float)(0)));
    }
  }
}

static void cpu_mm(__half* c, const __half* a, bool transpose_a, const __half* b, bool transpose_b,
                   float beta, int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        int ai = transpose_a ? kk * m + i : i * k + kk;
        int bi = transpose_b ? j * k + kk : kk * n + j;
        sum += a[ai] * b[bi];
      }
      c[i * n + j] = static_cast<half>(beta * static_cast<float>(c[i * n + j]) + sum);
    }
  }
}

static void cpu_add_bias_and_re(__half* top, __half* middle, const __half* bias, bool is_relu,
                                int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      __half t = top[i * n + j] + bias[j];
      middle[i * n + j] = t;
      if (is_relu)
        top[i * n + j] = t < 0 ? __float2half(0.0f) : t;
      else
        top[i * n + j] = t;
    }
  }
}

static void cpu_reverse_add_bias_and_re(__half* bias_grad, __half* dRelu, __half* middle,
                                        const __half* bprop_out, int m, int n, bool is_tail) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if ((middle[i * n + j] <= 0 && is_tail) || (middle[i * n + j] < 0 && !is_tail)) {
        dRelu[i * n + j] = 0.0f;
      } else {
        dRelu[i * n + j] = bprop_out[i * n + j];
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < m; ++j) sum += dRelu[j * n + i];
    bias_grad[i] = sum;
  }
}

static float compare_bit_array(const __half* arr1, const __half* arr2, size_t n, float threshold,
                               bool is_print) {
  size_t m = 0;
  for (size_t i = 0; i < n; i++) {
    int i_bit = i / 16;
    int j_bit = i % 16;
    int bit_val = (int)arr2[i / 16];
    int val2 = (bit_val >> j_bit) & 1;
    int val1 = (int)arr1[i];
    // if (val1 != val2) HCTR_LOG(INFO, WORLD, "%d, %d, %d\n", (int)i, val1, val2);
    // bool val = int(arr2[i / 8]) << (i % 8);
  }
  return m;
}

static float compare_array(const __half* arr1, const __half* arr2, size_t n, float threshold,
                           bool is_print) {
  size_t m = 0;
  for (size_t i = 0; i < n; i++) {
    // if(is_print && arr2[i] != 0.0) HCTR_LOG(INFO, WORLD, "%ld, %f, %f\n",i, (float)arr1[i],
    // (float)arr2[i]);
    if (isnan((float)arr1[i]) || isnan((float)arr2[i]) || isinf((float)arr1[i]) ||
        isinf((float)arr2[i])) {
      HCTR_LOG(INFO, WORLD, "Nan or Inf Error\n");
      return INT_MAX;
    }
    if (fabs(arr1[i] - arr2[i]) > threshold) {
      if (arr2[i] == 0 && fabs(arr1[i]) > threshold) {
        HCTR_LOG(INFO, WORLD, "%ld, %f, %f\n", i, (float)arr1[i], (float)arr2[i]);
        m++;
      } else if (fabs(arr1[i] - arr2[i]) / arr2[i] > threshold) {
        HCTR_LOG(INFO, WORLD, "%ld, %f, %f\n", i, (float)arr1[i], (float)arr2[i]);
        m++;
      }
    }
  }
  return 1.0f * m / n;
}

void set_l2_policy(const cudaStream_t& stream, __half* ptr, int num_bytes) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  size_t size = std::min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
  size_t window_size = std::min(prop.accessPolicyMaxWindowSize, num_bytes);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, window_size);

  cudaStreamAttrValue stream_attribute;
  stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr);
  stream_attribute.accessPolicyWindow.num_bytes = window_size;
  stream_attribute.accessPolicyWindow.hitRatio = 1.0;
  stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  HCTR_LOG(INFO, WORLD,
           "Stream: %p, ptr: %p, num_bytes: %d, window_size: %d, set-aside cache: %d\n", &stream,
           ptr, num_bytes, (int)window_size, (int)size);
}

__half **h_kernel, **h_kernel_grad, **h_bias_grad, **h_bias;
__half **h_bottom, **h_bottom_grad, **h_middle, **h_middle_grad, **h_top, **h_top_grad;
FusedReluBiasFullyConnectedLayer** fc_layers;
InteractionLayer<__half>* inter_layer;
std::vector<Layer*> layers;

static void init_data_cpu(uint32_t* input_dims, uint32_t* output_dims, uint32_t n_layers,
                          uint32_t batch_size) {
  h_kernel = new __half*[n_layers];
  h_kernel_grad = new __half*[n_layers];
  h_bias_grad = new __half*[n_layers];
  h_bias = new __half*[n_layers];
  for (uint32_t i = 0; i < n_layers; i++) {
    h_kernel[i] = new __half[input_dims[i] * output_dims[i]];
    h_kernel_grad[i] = new __half[input_dims[i] * output_dims[i]];
    h_bias_grad[i] = new __half[output_dims[i]];
    h_bias[i] = new __half[output_dims[i]];
    fill_data(h_kernel[i], input_dims[i] * output_dims[i]);
    fill_data(h_kernel_grad[i], input_dims[i] * output_dims[i]);
    fill_data(h_bias_grad[i], output_dims[i]);
    fill_data(h_bias[i], output_dims[i]);
  }
  h_bottom = new __half*[n_layers];
  h_bottom_grad = new __half*[n_layers];
  h_middle = new __half*[n_layers];
  h_middle_grad = new __half*[n_layers];
  h_top = new __half*[n_layers];
  h_top_grad = new __half*[n_layers];
  // Forward
  h_bottom[0] = new __half[batch_size * input_dims[0]];
  h_middle[0] = new __half[batch_size * output_dims[0]];
  h_top[0] = new __half[batch_size * output_dims[0]];
  fill_data(h_bottom[0], batch_size * input_dims[0]);
  fill_data(h_middle[0], batch_size * output_dims[0]);
  fill_data(h_top[0], batch_size * output_dims[0]);

  for (uint32_t i = 1; i < n_layers; i++) {
    h_bottom[i] = h_top[i - 1];
    h_middle[i] = new __half[batch_size * output_dims[i]];
    uint32_t tmp_dim = output_dims[i];
    if (i < n_layers - 1 && input_dims[i + 1] > tmp_dim) {
      tmp_dim = input_dims[i + 1];
    }
    h_top[i] = new __half[batch_size * tmp_dim];
    fill_data(h_middle[i], batch_size * output_dims[i]);
    fill_data(h_top[i], batch_size * output_dims[i]);
  }
  // Backward
  h_bottom_grad[n_layers - 1] = new __half[batch_size * input_dims[n_layers - 1]];
  h_middle_grad[n_layers - 1] = new __half[batch_size * output_dims[n_layers - 1]];
  h_top_grad[n_layers - 1] = new __half[batch_size * output_dims[n_layers - 1]];
  fill_data(h_top_grad[n_layers - 1], batch_size * output_dims[n_layers - 1]);
  fill_data(h_middle_grad[n_layers - 1], batch_size * output_dims[n_layers - 1]);
  fill_data(h_bottom_grad[n_layers - 1], batch_size * input_dims[n_layers - 1]);

  for (int i = n_layers - 2; i >= 0; i--) {
    h_top_grad[i] = h_bottom_grad[i + 1];
    h_middle_grad[i] = new __half[batch_size * output_dims[i]];
    h_bottom_grad[i] = new __half[batch_size * input_dims[i]];
    fill_data(h_middle_grad[i], batch_size * output_dims[i]);
    fill_data(h_bottom_grad[i], batch_size * input_dims[i]);
  }
}

static void copy_data_from_cpu(uint32_t* input_dims, uint32_t* output_dims, uint32_t n_layers,
                               uint32_t batch_size) {
  __half** d_kernel = new __half*[n_layers];
  __half** d_bias = new __half*[n_layers];
  __half** d_kernel_grad = new __half*[n_layers];
  __half** d_bias_grad = new __half*[n_layers];
  for (uint32_t i = 0; i < n_layers; i++) {
    d_kernel[i] = fc_layers[i]->get_weights_half_tensor()[0].get_ptr();
    d_bias[i] = fc_layers[i]->get_weights_half_tensor()[1].get_ptr();
    d_kernel_grad[i] = fc_layers[i]->get_weights_grad_tensor()[0].get_ptr();
    d_bias_grad[i] = fc_layers[i]->get_weights_grad_tensor()[1].get_ptr();
    HCTR_LIB_THROW(cudaMemcpy(d_kernel[i], h_kernel[i],
                              input_dims[i] * output_dims[i] * sizeof(__half),
                              cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(
        cudaMemcpy(d_bias[i], h_bias[i], output_dims[i] * sizeof(__half), cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(d_kernel_grad[i], h_kernel_grad[i],
                              input_dims[i] * output_dims[i] * sizeof(__half),
                              cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(d_bias_grad[i], h_bias_grad[i], output_dims[i] * sizeof(__half),
                              cudaMemcpyHostToDevice));
  }
}

static float check_data_cpu_and_gpu(__half* host, __half* device, uint32_t N, float threshold,
                                    bool is_bit = false, bool is_print = false) {
  __half* d2h = new __half[N];
  HCTR_LIB_THROW(cudaMemcpy(d2h, device, N * sizeof(__half), cudaMemcpyDeviceToHost));
  if (is_bit)
    return compare_bit_array(host, d2h, N, threshold, is_print);
  else
    return compare_array(host, d2h, N, threshold, is_print);
}

static void group_dense_layer_test(uint32_t* input_dims, uint32_t* output_dims, int* head_body_tail,
                                   bool* is_relu, bool* fuse_wb, uint32_t n_out_dims,
                                   uint32_t batch_size, const ConfigSet& config_set) {
  uint32_t n_layers = n_out_dims;
  for (uint32_t i = 0; i < n_out_dims; i++) {
    HCTR_LOG(INFO, WORLD,
             "The %dth layer: batch size = %d, input dimension = %d, output dimension = %d\n",
             i + 1, batch_size, input_dims[i], output_dims[i]);
  }
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> blobs_buff =
      GeneralBuffer2<CudaAllocator>::create();
  std::shared_ptr<BufferBlock2<float>> master_weights_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> weights_buff = blobs_buff->create_block<__half>();
  std::shared_ptr<BufferBlock2<__half>> weights_grad_buff = blobs_buff->create_block<__half>();

  Tensor2<__half> train_in_tensor[n_out_dims], train_out_tensor[n_out_dims];
  Tensor2<__half> mask_in_tensor[n_out_dims], mask_out_tensor[n_out_dims];
  Tensor2<__half> dRelu_in_tensor[n_out_dims], dRelu_out_tensor[n_out_dims];
  Tensor2<__half> db_in_tensor[n_out_dims], db_out_tensor[n_out_dims];
  fc_layers = new FusedReluBiasFullyConnectedLayer*[n_out_dims];
  Activation_t relu[n_out_dims];
  for (uint32_t i = 0; i < n_out_dims; i++) {
    if (is_relu[i])
      relu[i] = Activation_t::Relu;
    else
      relu[i] = Activation_t::None;
  }
  std::shared_ptr<GPUResource> gpu_resource = test::get_default_gpu();
  // Head layer
  blobs_buff->reserve({batch_size, input_dims[0]}, &train_in_tensor[0]);
  blobs_buff->reserve({batch_size, input_dims[0]}, &mask_in_tensor[0]);
  blobs_buff->reserve({batch_size, output_dims[0]}, &train_out_tensor[0]);
  blobs_buff->reserve({batch_size, output_dims[0]}, &mask_out_tensor[0]);
  blobs_buff->reserve({batch_size, output_dims[0]}, &dRelu_out_tensor[0]);
  fc_layers[0] = new FusedReluBiasFullyConnectedLayer(
      master_weights_buff, weights_buff, weights_grad_buff, blobs_buff, train_in_tensor[0],
      mask_in_tensor[0], dRelu_in_tensor[0], db_in_tensor[0], train_out_tensor[0],
      mask_out_tensor[0], dRelu_out_tensor[0], db_out_tensor[0], gpu_resource, FcPosition_t::Head,
      relu[0], false, std::vector<Initializer_t>(), config_set.async_mlp_wgrad, false, fuse_wb[0]);
  layers.push_back(fc_layers[0]);
  // Body layer and Tail layer
  for (uint32_t i = 1; i < n_layers; i++) {
    // Interaction layer
    if (input_dims[i] != output_dims[i - 1]) {
      // int in_dims[8]  = {16,  512, 256, 480, 1024, 1024, 512, 256};
      // int out_dims[8] = {512, 256, 128, 1024, 1024, 512, 256, 1};
      Tensor2<__half> in_mlp_tensor = train_out_tensor[i - 1];  // 640x128
      Tensor2<__half> in_emb_tensor;                            // 640 x 26 x 128
      blobs_buff->reserve({batch_size, 26, output_dims[i - 1]}, &in_emb_tensor);

      Tensor2<__half> out_tensor, grad_tensor;

      // blobs_buff->reserve({batch_size, input_dims[i]}, &out_tensor); //640x480
      // blobs_buff->reserve({batch_size, input_dims[i]}, &grad_tensor); //640x480
      inter_layer =
          new InteractionLayer<__half>(in_mlp_tensor, in_emb_tensor, out_tensor, grad_tensor,
                                       blobs_buff, gpu_resource, true, false);
      layers.push_back(inter_layer);
      train_in_tensor[i] = out_tensor;
      blobs_buff->reserve({batch_size, input_dims[i]}, &mask_in_tensor[i]);
      // blobs_buff->reserve({batch_size, input_dims[i]}, &dRelu_in_tensor[i]);
      db_in_tensor[i] = grad_tensor;
      blobs_buff->reserve({batch_size, output_dims[i]}, &train_out_tensor[i]);
      blobs_buff->reserve({batch_size, output_dims[i]}, &mask_out_tensor[i]);
      blobs_buff->reserve({batch_size, output_dims[i]}, &dRelu_out_tensor[i]);
    } else {
      train_in_tensor[i] = train_out_tensor[i - 1];
      mask_in_tensor[i] = mask_out_tensor[i - 1];
      dRelu_in_tensor[i] = dRelu_out_tensor[i - 1];
      db_in_tensor[i] = db_out_tensor[i - 1];
      blobs_buff->reserve({batch_size, output_dims[i]}, &train_out_tensor[i]);
      blobs_buff->reserve({batch_size, output_dims[i]}, &mask_out_tensor[i]);
      blobs_buff->reserve({batch_size, output_dims[i]}, &dRelu_out_tensor[i]);
    }

    if (2 == head_body_tail[i]) {  // tail
      fc_layers[i] = new FusedReluBiasFullyConnectedLayer(
          master_weights_buff, weights_buff, weights_grad_buff, blobs_buff, train_in_tensor[i],
          mask_in_tensor[i], dRelu_in_tensor[i], db_in_tensor[i], train_out_tensor[i],
          mask_out_tensor[i], dRelu_out_tensor[i], db_out_tensor[i], gpu_resource,
          FcPosition_t::Tail, relu[i], false, std::vector<Initializer_t>(),
          config_set.async_mlp_wgrad, false, fuse_wb[i]);
    } else if (1 == head_body_tail[i]) {  // body
      fc_layers[i] = new FusedReluBiasFullyConnectedLayer(
          master_weights_buff, weights_buff, weights_grad_buff, blobs_buff, train_in_tensor[i],
          mask_in_tensor[i], dRelu_in_tensor[i], db_in_tensor[i], train_out_tensor[i],
          mask_out_tensor[i], dRelu_out_tensor[i], db_out_tensor[i], gpu_resource,
          FcPosition_t::Body, relu[i], false, std::vector<Initializer_t>(),
          config_set.async_mlp_wgrad, false, fuse_wb[i]);
    } else {  // head
      fc_layers[i] = new FusedReluBiasFullyConnectedLayer(
          master_weights_buff, weights_buff, weights_grad_buff, blobs_buff, train_in_tensor[i],
          mask_in_tensor[i], dRelu_in_tensor[i], db_in_tensor[i], train_out_tensor[i],
          mask_out_tensor[i], dRelu_out_tensor[i], db_out_tensor[i], gpu_resource,
          FcPosition_t::Head, relu[i], false, std::vector<Initializer_t>(),
          config_set.async_mlp_wgrad, false, fuse_wb[i]);
    }
    layers.push_back(fc_layers[i]);
  }
  // Initialize tensors to 0 and choose cublas algorithms
  blobs_buff->allocate();
  for (uint32_t i = 0; i < n_layers; i++) {
    fc_layers[i]->initialize();
    printf("------------------------layers = %d---------------------------\n", i);
    fc_layers[i]->search_algorithm();
  }
  // Reset tensors to 0 to ensure all the data are the same as original utest(clear the side effect
  // of optimize)
  Tensor2<__half> weights = weights_buff->as_tensor();
  Tensor2<__half> weights_grad = weights_grad_buff->as_tensor();
  HCTR_LIB_THROW(cudaMemset(weights.get_ptr(), 0, weights.get_size_in_bytes()));
  HCTR_LIB_THROW(cudaMemset(weights_grad.get_ptr(), 0, weights_grad.get_size_in_bytes()));

  HCTR_LIB_THROW(cudaDeviceSynchronize());

  init_data_cpu(input_dims, output_dims, n_layers, batch_size);
  copy_data_from_cpu(input_dims, output_dims, n_layers, batch_size);

  // check if grad and bias are equal
  __half** d_kernel = new __half*[n_layers];
  __half** d_bias = new __half*[n_layers];
  __half** d_kernel_grad = new __half*[n_layers];
  __half** d_bias_grad = new __half*[n_layers];
  for (uint32_t i = 0; i < n_layers; i++) {
    d_kernel[i] = fc_layers[i]->get_weights_half_tensor()[0].get_ptr();
    d_bias[i] = fc_layers[i]->get_weights_half_tensor()[1].get_ptr();
    d_kernel_grad[i] = fc_layers[i]->get_weights_grad_tensor()[0].get_ptr();
    d_bias_grad[i] = fc_layers[i]->get_weights_grad_tensor()[1].get_ptr();
    ASSERT_LT(
        check_data_cpu_and_gpu(h_kernel[i], d_kernel[i], input_dims[i] * output_dims[i], 1e-3),
        0.05)
        << "kernel cross_check result fail" << std::endl;
    ASSERT_LT(check_data_cpu_and_gpu(h_bias[i], d_bias[i], output_dims[i], 1e-3), 0.05)
        << "bias cross_check result fail" << std::endl;
    ASSERT_LT(check_data_cpu_and_gpu(h_kernel_grad[i], d_kernel_grad[i],
                                     input_dims[i] * output_dims[i], 1e-3),
              0.05)
        << "kernel_grad cross_check result fail" << std::endl;
    ASSERT_LT(check_data_cpu_and_gpu(h_bias_grad[i], d_bias_grad[i], output_dims[i], 1e-3), 0.05)
        << "bias_grad cross_check result fail" << std::endl;
  }

  // initialize X
  HCTR_LIB_THROW(cudaMemcpy(train_in_tensor[0].get_ptr(), h_bottom[0],
                            sizeof(__half) * batch_size * input_dims[0], cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  // Forward pass (CPU)
  if (!config_set.is_perf_test) {
    for (uint32_t i = 0; i < n_layers; i++) {
      cpu_mm(h_top[i], h_bottom[i], false, h_kernel[i], false, 0.0, batch_size, input_dims[i],
             output_dims[i]);
      cpu_add_bias_and_re(h_top[i], h_middle[i], h_bias[i], is_relu[i], batch_size, output_dims[i]);
    }

    // Forward pass (GPU)
    HCTR_LIB_THROW(cudaDeviceSynchronize());
    for (uint32_t i = 0; i < n_layers; i++) {
      if (i > 0 && input_dims[i] != output_dims[i - 1]) {
        HCTR_LIB_THROW(cudaMemcpy(train_in_tensor[i].get_ptr(), h_bottom[i],
                                  sizeof(__half) * batch_size * input_dims[i],
                                  cudaMemcpyHostToDevice));
      }
      HCTR_LIB_THROW(cudaDeviceSynchronize());
      fc_layers[i]->fprop(true);
    }
    HCTR_LIB_THROW(cudaDeviceSynchronize());

    // Check results
    for (uint32_t i = 0; i < n_layers; i++) {
      ASSERT_LE(check_data_cpu_and_gpu(h_top[i], train_out_tensor[i].get_ptr(),
                                       batch_size * output_dims[i], 1e-4),
                0.0)
          << "Forward, Y of the " << i << "th layer cross_check result fail" << std::endl;
    }
    // initialize dX
    HCTR_LIB_THROW(cudaMemcpy(mask_out_tensor[n_layers - 1].get_ptr(), h_middle[n_layers - 1],
                              sizeof(__half) * batch_size * output_dims[n_layers - 1],
                              cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(train_out_tensor[n_layers - 1].get_ptr(), h_top_grad[n_layers - 1],
                              sizeof(__half) * batch_size * output_dims[n_layers - 1],
                              cudaMemcpyHostToDevice));

    HCTR_LIB_THROW(cudaDeviceSynchronize());
    // Backward pass (CPU)
    for (int i = n_layers - 1; i >= 0; i--) {
      if (!is_relu[i]) {
        memcpy(h_middle[i], h_top_grad[i], batch_size * output_dims[i] * sizeof(__half));
        for (uint32_t col = 0; col < output_dims[i]; col++) {
          float sum = 0.0;
          for (uint32_t row = 0; row < batch_size; row++) {
            sum = sum + h_top_grad[i][row * output_dims[i] + col];
          }
          h_bias_grad[i][col] = sum;
        }
      } else {
        cpu_reverse_add_bias_and_re(h_bias_grad[i], h_middle[i], h_middle[i], h_top_grad[i],
                                    batch_size, output_dims[i], i == int(n_layers - 1));
      }
      cpu_mm(h_kernel_grad[i], h_bottom[i], true, h_middle[i], false, 1.0, input_dims[i],
             batch_size, output_dims[i]);
      cpu_mm(h_bottom_grad[i], h_middle[i], false, h_kernel[i], true, 0.0, batch_size,
             output_dims[i], input_dims[i]);
    }

    // Backward pass (GPU)
    HCTR_LIB_THROW(cudaDeviceSynchronize());
    for (int i = n_layers - 1; i >= 0; i--) {
      fc_layers[i]->bprop();
    }
    HCTR_LIB_THROW(cudaDeviceSynchronize());

    // Check results
    for (int i = n_layers - 1; i >= 0; i--) {
      ASSERT_LE(
          check_data_cpu_and_gpu(h_bias_grad[i], db_out_tensor[i].get_ptr(), output_dims[i], 1e-3),
          0.0)
          << "Backward, dBias of the " << i << "th layer cross_check result fail" << std::endl;
      ASSERT_LE(check_data_cpu_and_gpu(h_kernel_grad[i],
                                       fc_layers[i]->get_weights_grad_tensor()[0].get_ptr(),
                                       input_dims[i] * output_dims[i], 1e-3),
                0.0)
          << "Backward, dW of the " << i << "th layer cross_check result fail" << std::endl;
      if (i > 0 && input_dims[i] != output_dims[i - 1]) {
        break;
      }
    }
    // If async_mlp_wgrad is true, then here dX is stored in mask_in_tensor[0] rather than
    // train_in_tensor[0]
    // ASSERT_LE(check_data_cpu_and_gpu(h_bottom_grad[0], train_in_tensor[0].get_ptr(),
    //                                  batch_size * input_dims[0], 1e-3),
    //           0.0)
    //     << "Backward, dX of the " << 0 << "th layer cross_check result fail" << endl;
  } else {
    float time_fprop = 0.0, time_bprop = 0.0;
    int layer_start = 0;
    int layer_end = (int)layers.size();
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool graph_inited = false;
    std::string nvtx_str;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (size_t test_loop_idx = config_set.test_loop_cnt; test_loop_idx > 0; --test_loop_idx) {
      printf("test_loop_idx = %ld\n", test_loop_idx);

      if (test_loop_idx == 1) {
        if (config_set.use_cuda_graph && !graph_inited) {
          cudaStreamBeginCapture(gpu_resource->get_stream(), cudaStreamCaptureModeThreadLocal);
        } else {
          cudaProfilerStart();
        }
      }

      for (int fprop_idx = layer_start; fprop_idx < layer_end; ++fprop_idx) {
        // printf("fprop_idx = %d, layers = 0x%lx\n", fprop_idx, (size_t)layers[fprop_idx]);
        layers[fprop_idx]->fprop(true);
      }

      for (int bprop_idx = layer_end - 1; bprop_idx >= layer_start; --bprop_idx) {
        if (config_set.use_nvtx) {
          nvtx_str = std::to_string(bprop_idx);
          nvtxRangePush(nvtx_str.c_str());
        }
        for (size_t layer_loop_idx = 0; layer_loop_idx < config_set.layer_loop_cnt;
             ++layer_loop_idx) {
          layers[bprop_idx]->bprop();
        }
        if (config_set.use_nvtx) nvtxRangePop();
      }
      if (test_loop_idx == 1) {
        if (config_set.use_cuda_graph && !graph_inited) {
          cudaStreamEndCapture(gpu_resource->get_stream(), &graph);
          cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
          graph_inited = true;
        } else {
          cudaProfilerStop();
        }
      }
    }

    float mean_time = 0.0f;
    // size_t test_loop = config_set.test_loop_cnt;
    size_t test_loop = 10000;
    for (size_t test_loop_idx = 0; test_loop_idx < test_loop; ++test_loop_idx) {
      if (graph_inited) {
        if (0 == test_loop_idx) {
          cudaProfilerStart();
        }
        float elapsedTime = 0.0f;
        cudaEventRecord(start, gpu_resource->get_stream());
        cudaGraphLaunch(graph_exec, gpu_resource->get_stream());
        cudaEventRecord(stop, gpu_resource->get_stream());
        cudaStreamSynchronize(gpu_resource->get_stream());
        cudaEventElapsedTime(&elapsedTime, start, stop);
        if (test_loop_idx % 1000 == 0) {
          printf("test_loop_idx = %ld, elapsed_time = %f\n", test_loop_idx, elapsedTime);
        }
        mean_time += elapsedTime;
        if (10 == test_loop_idx) {
          cudaProfilerStop();
        }
      }
    }
    printf("test_loop = %ld, elapsed_time = %f\n", test_loop, mean_time / test_loop);
  }
}
}  // namespace

// ConfigSet config_set = {false, true, true, 10, 1000};
ConfigSet config_set = {true, true, false, true, true, 10, 1};
/*
int out_dims_top[5] = {1024, 1024, 512, 256, 1};
bool is_relu_top[5] = {true, true, true, true, false};
TEST(group_dense_layer_test, top) { group_dense_layer_test(out_dims_top, is_relu_top, 5, 640, 480,
config_set); }; int out_dims_bottom[3] = {512, 256, 128}; bool is_relu_bottom[3] = {true, true,
true}; TEST(group_dense_layer_test, bottom) { group_dense_layer_test(out_dims_bottom,
is_relu_bottom, 3, 640, 16, config_set); };
*/
uint32_t in_dims[8] = {16, 512, 256, 480, 1024, 1024, 512, 256};
uint32_t out_dims[8] = {512, 256, 128, 1024, 1024, 512, 256, 1};
int head_body_tail[8] = {0, 1, 2, 0, 1, 1, 1, 2};
bool is_relu[9] = {true, true, true, true, true, true, true, false};
bool fuse_wb[8] = {false, false, false, true, true, true, true, true};
TEST(group_dense_layer_test, all) {
  group_dense_layer_test(in_dims, out_dims, head_body_tail, is_relu, fuse_wb, 8, 640, config_set);
};

// int batch_sizes[8] = {448, 480, 500, 512, 560, 576, 640, 6912};
// TEST(group_dense_layer_test, bottom) { group_dense_layer_test_batches(out_dims_bot, is_relu_bot,
// 3, batch_sizes, 8, 16, true); }; TEST(group_dense_layer_test, top) {
// group_dense_layer_test_batches(out_dims_top, is_relu_top, 5, batch_sizes, 8, 16, true); };

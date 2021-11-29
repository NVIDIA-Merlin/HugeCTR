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

#include "HugeCTR/include/layers/fused_relu_bias_fully_connected_layer.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <vector>
#include <sys/time.h>
#include "cublas_v2.h"
#include "gtest/gtest.h"
#include "utest/test_utils.h"
#include "nvToolsExt.h"
#include <cuda_profiler_api.h>

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

using namespace std;
using namespace HugeCTR;

#define uint unsigned int

#define REPEAT_NUM 100

static void fill_data(__half* data, int N) {
  unsigned seed = time(0);
  srand(seed);
  for(int i=0;i<N;i++) {
    data[i] =(__half((float) (rand() % 3 - 1)));

    if (rand() % 50) {
        data[i] = (__half((float) (0)));
    }
    //data[i] =(__half((float) (rand()%100-50)/200));
    //data[i] = (__half(sign_list[rand()%3]));
    //data[i] = (__half(sign_list[rand()%3]*pow(2, -rand()%10)));
    //printf("%d,%f\n",i,(float)data[i]);
  }
}
static void cpu_mm(__half *c, const __half *a, bool transpose_a, const __half *b, bool transpose_b,
                   float beta, int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        int ai = transpose_a ? kk * m + i : i * k + kk;
        int bi = transpose_b ? j * k + kk : kk * n + j;
        //sum = fmaf(__half2float(a[ai]) , __half2float(b[bi]), sum);
        sum += a[ai]*b[bi];
      }
      c[i * n + j] = static_cast<half>(beta * static_cast<float>(c[i * n +j]) + sum);
      //c[i * n + j] = beta * c[i * n +j] + __float2half(sum);
    }
  }
}

static void cpu_add_bias_and_re(__half *top, __half *middle, const __half *bias, bool is_relu, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      __half t = top[i * n + j] + bias[j];
      middle[i * n + j] = t;
      if (is_relu) top[i * n + j] = t < 0 ? __float2half(0.0f) : t;
      else top[i * n + j]  = t;
    }
  }
}

static void cpu_reverse_add_bias_and_re(__half *bias_grad, __half *dRelu, __half *middle, const __half *bprop_out,
                                        int m, int n, bool is_tail) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      if ((middle[i * n + j] <= 0 && is_tail) || (middle[i * n + j] < 0 && !is_tail)) {
        dRelu[i * n + j] = 0.0f;
      } else {
        dRelu[i * n + j] = bprop_out[i * n + j];
      }
    }

  for (int i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < m; ++j) sum += dRelu[j * n + i];
    bias_grad[i] = sum;
  }
}

static float compare_bit_array(const __half *arr1, const __half *arr2, size_t n, float threshold, bool is_print) {
  size_t m = 0;
  for (size_t i = 0; i < n; i++) {
    int i_bit = i / 16;
    int j_bit = i % 16;
    int bit_val = (int)arr2[i/16];
    int val2 = (bit_val >> j_bit) & 1;
    int val1 = (int)arr1[i];
    //if (val1 != val2) printf("%d, %d, %d\n", (int)i, val1, val2);
   // bool val = int(arr2[i / 8]) << (i % 8);
  }
  return m;
}

static float compare_array(const __half *arr1, const __half *arr2, size_t n, float threshold, bool is_print) {
  size_t m = 0;
  for (size_t i = 0; i < n; i++) {
             //if(is_print && arr2[i] != 0.0) printf("%ld, %f, %f\n",i, (float)arr1[i], (float)arr2[i]);
    if (isnan((float)arr1[i]) || isnan((float)arr2[i])
      || isinf((float)arr1[i]) || isinf((float)arr2[i])) {
            printf("Nan or Inf Error\n");
            return INT_MAX;
    }
    if (fabs(arr1[i] - arr2[i]) > threshold) {
      if(arr2[i] == 0 && fabs(arr1[i]) > threshold) {
              printf("%ld, %f, %f\n",i, (float)arr1[i], (float)arr2[i]);
              m++;
      } else if(fabs(arr1[i]-arr2[i])/arr2[i] > threshold) {
              printf("%ld, %f, %f\n",i, (float)arr1[i], (float)arr2[i]);
              m++;
      }
    }
  }
  return 1.0f * m / n;
}

void set_l2_policy(const cudaStream_t &stream, __half* ptr, int num_bytes) {
  cudaDeviceProp prop;                                                               
  cudaGetDeviceProperties(&prop, 0);                                        
  size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
  //cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                         
  
  size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes); 
  cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, window_size);

  cudaStreamAttrValue stream_attribute; 
  stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);
  stream_attribute.accessPolicyWindow.num_bytes = window_size;
  stream_attribute.accessPolicyWindow.hitRatio  = 1.0;  
  stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
  stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  printf("Stream: %p, ptr: %p, num_bytes: %d, window_size: %d, set-aside cache: %d\n", &stream, ptr, num_bytes, (int)window_size, (int)size);
}

__half **h_kernel, **h_kernel_grad, **h_bias_grad, **h_bias;
__half **h_bottom, **h_bottom_grad, **h_middle, **h_middle_grad, **h_top, **h_top_grad;
test::GaussianDataSimulator simulator(0.0f, 1.0f);
FusedReluBiasFullyConnectedLayer** fc_layers;

static void init_data_cpu(uint* input_dims, uint* output_dims, uint n_layers, uint batch_size) {
  h_kernel      = new __half*[n_layers];
  h_kernel_grad = new __half*[n_layers];
  h_bias_grad   = new __half*[n_layers];
  h_bias        = new __half*[n_layers];
  for (uint i=0;i<n_layers;i++) {
    h_kernel[i]      = new __half[input_dims[i]*output_dims[i]];
    h_kernel_grad[i] = new __half[input_dims[i]*output_dims[i]];
    h_bias_grad[i]   = new __half[output_dims[i]];
    h_bias[i]        = new __half[output_dims[i]];
    //simulator.fill(h_kernel[i],      input_dims[i]*output_dims[i]);
    //simulator.fill(h_kernel_grad[i], input_dims[i]*output_dims[i]);
    //simulator.fill(h_bias_grad[i],   output_dims[i]);
    //simulator.fill(h_bias[i],        output_dims[i]);
    fill_data(h_kernel[i],      input_dims[i]*output_dims[i]);
    fill_data(h_kernel_grad[i], input_dims[i]*output_dims[i]);
    fill_data(h_bias_grad[i],   output_dims[i]);
    fill_data(h_bias[i],        output_dims[i]);

  }
  h_bottom        = new __half*[n_layers];
  h_bottom_grad   = new __half*[n_layers];
  h_middle        = new __half*[n_layers];
  h_middle_grad   = new __half*[n_layers];
  h_top           = new __half*[n_layers];
  h_top_grad      = new __half*[n_layers];
  // Forward
  h_bottom[0] = new __half[batch_size*input_dims[0]];
  h_middle[0] = new __half[batch_size*output_dims[0]];
  h_top[0]    = new __half[batch_size*output_dims[0]];
  //simulator.fill(h_bottom[0], batch_size*input_dims[0]);
  //simulator.fill(h_middle[0], batch_size*output_dims[0]);
  //simulator.fill(h_top[0],    batch_size*output_dims[0]);
  fill_data(h_bottom[0], batch_size*input_dims[0]);
  fill_data(h_middle[0], batch_size*output_dims[0]);
  fill_data(h_top[0],    batch_size*output_dims[0]);

  for( uint i=1;i<n_layers;i++) {
    h_bottom[i] = h_top[i-1];
    h_middle[i] = new __half[batch_size*output_dims[i]];
    h_top[i]    = new __half[batch_size*output_dims[i]];
    //simulator.fill(h_middle[i], batch_size*output_dims[i]);
    //simulator.fill(h_top[i],    batch_size*output_dims[i]);
    fill_data(h_middle[i], batch_size*output_dims[i]);
    fill_data(h_top[i],    batch_size*output_dims[i]);

  }
  // Backward
  h_bottom_grad[n_layers-1] = new __half[batch_size*input_dims[n_layers-1]];
  h_middle_grad[n_layers-1] = new __half[batch_size*output_dims[n_layers-1]];
  h_top_grad[n_layers-1]    = new __half[batch_size*output_dims[n_layers-1]];
  //simulator.fill(h_top_grad[n_layers-1],    batch_size*output_dims[n_layers-1]);
  //simulator.fill(h_middle_grad[n_layers-1], batch_size*output_dims[n_layers-1]);
  //simulator.fill(h_bottom_grad[n_layers-1], batch_size*input_dims[n_layers-1]);
  fill_data(h_top_grad[n_layers-1],    batch_size*output_dims[n_layers-1]);
  fill_data(h_middle_grad[n_layers-1], batch_size*output_dims[n_layers-1]);
  fill_data(h_bottom_grad[n_layers-1], batch_size*input_dims[n_layers-1]);

  for( int i=n_layers-2;i>=0;i--) {
    h_top_grad[i]    = h_bottom_grad[i+1];
    h_middle_grad[i] = new __half[batch_size*output_dims[i]];
    h_bottom_grad[i] = new __half[batch_size*input_dims[i]];
    //simulator.fill(h_middle_grad[i], batch_size*output_dims[i]);
    //simulator.fill(h_bottom_grad[i], batch_size*input_dims[i]);
    fill_data(h_middle_grad[i], batch_size*output_dims[i]);
    fill_data(h_bottom_grad[i], batch_size*input_dims[i]);
  }

}

static void copy_data_from_cpu(uint* input_dims, uint* output_dims, uint n_layers, uint batch_size) {
  __half** d_kernel      = new __half*[n_layers];
  __half** d_bias        = new __half*[n_layers];
  __half** d_kernel_grad = new __half*[n_layers];
  __half** d_bias_grad   = new __half*[n_layers];
  for (uint i=0;i<n_layers;i++) {
    d_kernel[i]      = fc_layers[i]->get_weights_half_tensor()[0].get_ptr();
    d_bias[i]        = fc_layers[i]->get_weights_half_tensor()[1].get_ptr();
    d_kernel_grad[i] = fc_layers[i]->get_weights_grad_tensor()[0].get_ptr();
    d_bias_grad[i]   = fc_layers[i]->get_weights_grad_tensor()[1].get_ptr();
    CK_CUDA_THROW_(cudaMemcpy(d_kernel[i],      h_kernel[i],
                            input_dims[i]*output_dims[i]*sizeof(__half), cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy(d_bias[i],        h_bias[i],
                            output_dims[i]*sizeof(__half),               cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy(d_kernel_grad[i], h_kernel_grad[i],
                            input_dims[i]*output_dims[i]*sizeof(__half), cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy(d_bias_grad[i],   h_bias_grad[i],
                            output_dims[i]*sizeof(__half),               cudaMemcpyHostToDevice));
  }
}

static float check_data_cpu_and_gpu(__half* host, __half* device, uint N, float threshold, bool is_bit = false, bool is_print = false) {
  __half* d2h = new __half[N];
  CK_CUDA_THROW_(cudaMemcpy(d2h, device, N*sizeof(__half), cudaMemcpyDeviceToHost));
  if (is_bit) return compare_bit_array(host, d2h, N, threshold, is_print);
  else return compare_array(host, d2h, N, threshold, is_print);
}

static void mlp_test(int* out_dims, bool* is_relu, uint n_out_dims, uint batch_size, uint input_dim, bool perf_test = false) {
  uint n_layers = n_out_dims;
  uint* input_dims  = new uint[n_out_dims];
  uint* output_dims = new uint[n_out_dims];
  input_dims[0] = input_dim;
  for (uint i=0;i<n_out_dims-1;i++) {
    input_dims[i+1] = out_dims[i];
    output_dims[i]  = out_dims[i];
  }
  output_dims[n_out_dims-1] = out_dims[n_out_dims-1];
  for (uint i=0;i<n_out_dims;i++)
  {
    printf("The %dth layer: batch size = %d, input dimension = %d, output dimension = %d\n",
                    i+1, batch_size, input_dims[i], output_dims[i]);
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
  for (uint i=0;i<n_out_dims;i++) {
    if (is_relu[i]) relu[i] = Activation_t::Relu;
    else relu[i] = Activation_t::None;
  }

  std::shared_ptr<GPUResource> gpu_resource = test::get_default_gpu();

  // Head layer
  blobs_buff->reserve({batch_size, input_dims[0]}, &train_in_tensor[0]);
  blobs_buff->reserve({batch_size, input_dims[0]}, &mask_in_tensor[0]);
  blobs_buff->reserve({batch_size, output_dims[0]}, &train_out_tensor[0]);
  blobs_buff->reserve({batch_size, output_dims[0]}, &mask_out_tensor[0]);
  blobs_buff->reserve({batch_size, output_dims[0]}, &dRelu_out_tensor[0]);
  // blobs_buff->reserve({1,          output_dims[0]}, &db_out_tensor[0]);
  fc_layers[0] = new FusedReluBiasFullyConnectedLayer(
      master_weights_buff, weights_buff, weights_grad_buff, blobs_buff, train_in_tensor[0],
      mask_in_tensor[0], dRelu_in_tensor[0], db_in_tensor[0], train_out_tensor[0], mask_out_tensor[0],
      dRelu_out_tensor[0], db_out_tensor[0], gpu_resource, FcPosition_t::Head,
      relu[0], false);

  // Body layer and Tail layer
  for ( uint i=1;i<n_layers;i++) {
    train_in_tensor[i] = train_out_tensor[i-1];
    mask_in_tensor[i]  = mask_out_tensor[i-1];
    dRelu_in_tensor[i] = dRelu_out_tensor[i-1];
    db_in_tensor[i]    = db_out_tensor[i-1];
    blobs_buff->reserve({batch_size, output_dims[i]}, &train_out_tensor[i]);
    blobs_buff->reserve({batch_size, output_dims[i]}, &mask_out_tensor[i]);
    blobs_buff->reserve({batch_size, output_dims[i]}, &dRelu_out_tensor[i]);
    // blobs_buff->reserve({1,          output_dims[i]}, &db_out_tensor[i]);
    if (i == n_layers-1) {
      fc_layers[i] = new FusedReluBiasFullyConnectedLayer(
          master_weights_buff, weights_buff, weights_grad_buff, blobs_buff, train_in_tensor[i],
          mask_in_tensor[i], dRelu_in_tensor[i], db_in_tensor[i], train_out_tensor[i], mask_out_tensor[i],
          dRelu_out_tensor[i], db_out_tensor[i], gpu_resource, FcPosition_t::Tail,
          relu[i], false);
    } else {
      fc_layers[i] = new FusedReluBiasFullyConnectedLayer(
          master_weights_buff, weights_buff, weights_grad_buff, blobs_buff, train_in_tensor[i],
          mask_in_tensor[i], dRelu_in_tensor[i], db_in_tensor[i], train_out_tensor[i], mask_out_tensor[i],
          dRelu_out_tensor[i], db_out_tensor[i], gpu_resource, FcPosition_t::Body,
          relu[i], false);
    }
  }

  // Initialize tensors to 0 and choose cublas algorithms
  blobs_buff->allocate();
  for (uint i=0;i<n_layers;i++) {
    fc_layers[i]->initialize();
    fc_layers[i]->search_algorithm();
  }
  // Reset tensors to 0 to ensure all the data are the same as original utest(clear the side effect
  // of optimize)

  Tensor2<__half> weights = weights_buff->as_tensor();
  Tensor2<__half> weights_grad = weights_grad_buff->as_tensor();
  CK_CUDA_THROW_(cudaMemset(weights.get_ptr(), 0, weights.get_size_in_bytes()));
  CK_CUDA_THROW_(cudaMemset(weights_grad.get_ptr(), 0, weights_grad.get_size_in_bytes()));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  init_data_cpu(input_dims, output_dims, n_layers, batch_size);
  copy_data_from_cpu(input_dims, output_dims, n_layers, batch_size);

  // check if grad and bias are equal
  __half** d_kernel      = new __half*[n_layers];
  __half** d_bias        = new __half*[n_layers];
  __half** d_kernel_grad = new __half*[n_layers];
  __half** d_bias_grad   = new __half*[n_layers];
  for (uint i=0;i<n_layers;i++) {
    d_kernel[i]      = fc_layers[i]->get_weights_half_tensor()[0].get_ptr();
    d_bias[i]        = fc_layers[i]->get_weights_half_tensor()[1].get_ptr();
    d_kernel_grad[i] = fc_layers[i]->get_weights_grad_tensor()[0].get_ptr();
    d_bias_grad[i]   = fc_layers[i]->get_weights_grad_tensor()[1].get_ptr();
    ASSERT_LT(check_data_cpu_and_gpu(h_kernel[i],      d_kernel[i],      input_dims[i]*output_dims[i], 1e-3), 0.05)
        << "kernel cross_check result fail" << endl;
    ASSERT_LT(check_data_cpu_and_gpu(h_bias[i],        d_bias[i],        output_dims[i], 1e-3), 0.05)
        << "bias cross_check result fail" << endl;
    ASSERT_LT(check_data_cpu_and_gpu(h_kernel_grad[i], d_kernel_grad[i], input_dims[i]*output_dims[i], 1e-3), 0.05)
        << "kernel_grad cross_check result fail" << endl;
    ASSERT_LT(check_data_cpu_and_gpu(h_bias_grad[i],   d_bias_grad[i],   output_dims[i], 1e-3), 0.05)
        << "bias_grad cross_check result fail" << endl;
  }

  // initialize X
  CK_CUDA_THROW_(
      cudaMemcpy(train_in_tensor[0].get_ptr(), h_bottom[0], sizeof(__half) * batch_size * input_dims[0], cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  // Forward pass (CPU)
  if (!perf_test) {
    for (uint i=0;i<n_layers;i++) {
      cpu_mm(h_top[i], h_bottom[i], false, h_kernel[i], false, 0.0, batch_size, input_dims[i], output_dims[i]);
      cpu_add_bias_and_re(h_top[i], h_middle[i], h_bias[i], is_relu[i], batch_size, output_dims[i]);
    }

   // Forward pass (GPU)
    CK_CUDA_THROW_(cudaDeviceSynchronize());
    for (uint i=0;i<n_layers;i++) {
      fc_layers[i]->fprop(true);
    }
    CK_CUDA_THROW_(cudaDeviceSynchronize());
  
  
    // Check results
    for (uint i=0;i<n_layers;i++) {
  //    ASSERT_LT(check_data_cpu_and_gpu(h_bottom[i], train_in_tensor[i].get_ptr(), batch_size*input_dims[i], 1e-3), 0.1)
  //          << "X of the "<<i<<"th layer cross_check result fail"<<endl;
  //    ASSERT_LT(check_data_cpu_and_gpu(h_kernel[i], fc_layers[i]->get_weights_half_tensor()[0].get_ptr(), input_dims[i]*output_dims[i], 1e-3), 0.05)
  //          << "W of the "<<i<<"th layer cross_check result fail"<<endl;
      ASSERT_LE(check_data_cpu_and_gpu(h_top[i], train_out_tensor[i].get_ptr(), batch_size*output_dims[i], 1e-4), 0.0)
              << "Y of the "<<i<<"th layer cross_check result fail"<<endl;
  //    ASSERT_LT(check_data_cpu_and_gpu(h_top[i], mask_out_tensor[i].get_ptr(), batch_size*output_dims[i], 1e-3, true), 0.0)
  //          << "mask of the "<<i<<"th layer cross_check result fail"<<endl;
  
    }
  
  
    // initialize dX
    CK_CUDA_THROW_(
        cudaMemcpy(mask_out_tensor[n_layers-1].get_ptr(), h_middle[n_layers-1], sizeof(__half) * batch_size * output_dims[n_layers-1], cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(
        cudaMemcpy(train_out_tensor[n_layers-1].get_ptr(), h_top_grad[n_layers-1], sizeof(__half) * batch_size * output_dims[n_layers-1], cudaMemcpyHostToDevice));
  //  for (uint i=0;i<n_layers;i++) {
  //    CK_CUDA_THROW_(
  //        cudaMemcpy(train_in_tensor[i].get_ptr(), h_bottom[i], sizeof(__half) * batch_size * input_dims[i], cudaMemcpyHostToDevice));
  //  }
  
    CK_CUDA_THROW_(cudaDeviceSynchronize());
    // Forward pass (CPU)
    for (int i=n_layers-1;i>=0;i--) {
      if (!is_relu[i]) {
        memcpy(h_middle[i], h_top_grad[i], batch_size*output_dims[i]*sizeof(__half));
        for (uint col=0; col<output_dims[i]; col++) {
  	float sum = 0.0;
  	for (uint row=0; row<batch_size; row++) {
  	  sum = sum + h_top_grad[i][row*output_dims[i]+col];
  	}
  	h_bias_grad[i][col] = sum;
        }
      }
      else cpu_reverse_add_bias_and_re(h_bias_grad[i], h_middle[i], h_middle[i], h_top_grad[i], batch_size, output_dims[i], i==int(n_layers-1) );
  
      cpu_mm(h_kernel_grad[i], h_bottom[i], true, h_middle[i], false, 1.0, input_dims[i], batch_size, output_dims[i]);
      cpu_mm(h_bottom_grad[i], h_middle[i], false, h_kernel[i], true, 0.0, batch_size, output_dims[i], input_dims[i]);
    }
  
   // Forward pass (GPU)
    CK_CUDA_THROW_(cudaDeviceSynchronize());
    for (int i=n_layers-1;i>=0;i--) {
      fc_layers[i]->bprop();
    }
    CK_CUDA_THROW_(cudaDeviceSynchronize());
  
  
    // Check results
    for (int i=n_layers-1;i>=0;i--) {
  //    ASSERT_LT(check_data_cpu_and_gpu(h_middle[i], dRelu_out_tensor[i].get_ptr(), batch_size*output_dims[1], 1e-3), 0.15)
  //          << "dRelu_out of the "<<i<<"th layer cross_check result fail"<<endl;
      ASSERT_LE(check_data_cpu_and_gpu(h_bias_grad[i], db_out_tensor[i].get_ptr(), output_dims[i], 1e-3), 0.0)
            << "dBias of the "<<i<<"th layer cross_check result fail"<<endl;
      ASSERT_LE(check_data_cpu_and_gpu(h_kernel_grad[i], fc_layers[i]->get_weights_grad_tensor()[0].get_ptr(), input_dims[i]*output_dims[i], 1e-3), 0.0)
              << "dW of the "<<i<<"th layer cross_check result fail"<<endl;
    }
    // If async_mlp_wgrad is true, then here dX is stored in mask_in_tensor[0] rather than train_in_tensor[0]
    ASSERT_LE(check_data_cpu_and_gpu(h_bottom_grad[0], train_in_tensor[0].get_ptr(), batch_size*input_dims[0], 1e-3), 0.0)
              << "dX of the "<<0<<"th layer cross_check result fail"<<endl;
  } else {
    cudaProfilerStart();
    int idx = 0;
    struct timeval ts1, ts2, ts3;
    float time_fprop=0.0, time_bprop=0.0;
    int layer_start = max(0, 0);
    int layer_end = min((int)n_layers, (int)n_layers);
    cudaEvent_t e1, e2, e3, e4;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);
    cudaEventCreate(&e4);
    int i_layer = 3;
    while (idx++<REPEAT_NUM) {
      CK_CUDA_THROW_(cudaDeviceSynchronize());
      //gettimeofday(&ts1, NULL);
      cudaEventRecord(e1, gpu_resource->get_stream());
      for (int i=layer_start;i<layer_end;i++) {
        //if (i==i_layer) cudaEventRecord(e1, gpu_resource->get_stream());
        fc_layers[i]->fprop(true);
        //if (i==i_layer) cudaEventRecord(e2, gpu_resource->get_stream());
      }
      cudaEventRecord(e2, gpu_resource->get_stream());
      cudaEventRecord(e3, gpu_resource->get_stream());
      //gettimeofday(&ts2, NULL);
      for (int i=layer_end-1;i>=layer_start;i--) {
        //if (i==i_layer) cudaEventRecord(e3, gpu_resource->get_stream());
        fc_layers[i]->bprop();
        //if (i==i_layer) cudaEventRecord(e4, gpu_resource->get_stream());
      }
      cudaEventRecord(e4, gpu_resource->get_stream());
      nvtxRangePop();
      cudaEventSynchronize(e4);
      //CK_CUDA_THROW_(cudaDeviceSynchronize());
      //gettimeofday(&ts3, NULL);
      float time_elapsed = 0.0;
      cudaEventElapsedTime(&time_elapsed, e1, e2);
      time_fprop += time_elapsed*1000;
      cudaEventElapsedTime(&time_elapsed, e3, e4);
      time_bprop += time_elapsed*1000;

      //float t_f=(ts2.tv_sec-ts1.tv_sec)*1000000+(ts2.tv_usec-ts1.tv_usec);
      //float t_b=(ts3.tv_sec-ts2.tv_sec)*1000000+(ts3.tv_usec-ts2.tv_usec);
      //time_fprop += time_elapsed;
      //time_bprop += t_b;
    }
    cudaProfilerStop();
    printf("time_fprop is %.10f\n",time_fprop/REPEAT_NUM);
    printf("time_bprop is %.10f\n",time_bprop/REPEAT_NUM);
  }


}

static void mlp_test_batches(int* out_dims, bool* is_relu, uint n_out_dims, int* batch_sizes, int nbatches, uint input_dim, bool perf_test = false) {
  for (int i=0;i<nbatches;i++) {
    mlp_test(out_dims, is_relu, n_out_dims, batch_sizes[i], input_dim, perf_test);
  }
}

//int out_dims_bot[3] = { 512, 256, 128 };
//bool is_relu_bot[3] = { true, true, true };
//TEST(mlp_test, bottom) { mlp_test(out_dims_bot, is_relu_bot, 3, 6912, 16, true); };
int out_dims_top[5] = { 1024, 1024, 512, 256, 1};
bool is_relu_top[5] = { true, true, true, true, false};
TEST(mlp_test, top) { mlp_test(out_dims_top, is_relu_top,  5, 640, 6912, true); };

int batch_sizes[8] = { 448, 480, 500, 512, 560, 576, 640, 6912};
//TEST(mlp_test, bottom) { mlp_test_batches(out_dims_bot, is_relu_bot, 3, batch_sizes, 8, 16, true); };
//TEST(mlp_test, top) { mlp_test_batches(out_dims_top, is_relu_top, 5, batch_sizes, 8, 16, true); };


/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/layers/reduce_sum_layer.hpp"

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-5f;

void reduce_sum_cpu(const float * input, 
                    float * output, 
                    int dim0,
                    int dim1,
                    int dim2, 
                    int axis) {
  if(axis == 0) {
    for(int j = 0; j < dim1; j++) {
      for(int k = 0; k < dim2; k++) {
        output[j*dim2+k] = 0;
        for(int i = 0; i < dim0; i++) {
          output[j*dim2+k] += input[i*dim1*dim2+j*dim2+k];
        }
      }
    }
  }
  else if(axis == 1) {
    for(int i = 0; i < dim0; i++) {
      for(int k = 0; k < dim2; k++) {
        output[i*dim2+k] = 0;
        for(int j = 0; j < dim1; j++) {
          output[i*dim2+k] += input[i*dim1*dim2+j*dim2+k];
        }
      }
    }
  }
  else if(axis == 2) {
    for(int i = 0; i < dim0; i++) {
      for(int j = 0; j < dim1; j++) {
        output[i*dim1+j] = 0;
        for(int k = 0; k < dim2; k++) {
          output[i*dim1+j] += input[i*dim1*dim2+j*dim2+k];
        }
      }
    }
  }
}

void reduce_sum_dgrad_cpu(const float * top_grad, 
                          float * dgrad, 
                          int dim0,
                          int dim1,
                          int dim2, 
                          int axis) {
  if(axis == 0) {
    for(int j = 0; j < dim1; j++) {
      for(int k = 0; k < dim2; k++) {
        for(int i = 0; i < dim0; i++) {
          dgrad[i*dim1*dim2+j*dim2+k] = top_grad[j*dim2+k];
        }
      }
    }
  }
  else if(axis == 1) {
    for(int i = 0; i < dim0; i++) {
      for(int k = 0; k < dim2; k++) {
        for(int j = 0; j < dim1; j++) {
          dgrad[i*dim1*dim2+j*dim2+k] = top_grad[i*dim2+k];
        }
      }
    }
  }
  else if(axis == 2) {
    for(int i = 0; i < dim0; i++) {
      for(int j = 0; j < dim1; j++) {
        for(int k = 0; k < dim2; k++) {
          dgrad[i*dim1*dim2+j*dim2+k] = top_grad[i*dim1+j];
        }
      }
    }
  }
}

void reduce_sum_test(int batch_size, int slot_num, int embedding_vec_size, int axis) {
  int dev_id = 0;
  std::shared_ptr<GeneralBuffer<float>> in_out_buf(new GeneralBuffer<float>());

  vector<int> in_dims = {batch_size, slot_num, embedding_vec_size};
  std::shared_ptr<Tensor<float>> in_tensor(new Tensor<float>(in_dims, in_out_buf, TensorFormat_t::HSW));
  std::shared_ptr<Tensor<float>> out_tensor;

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  ReduceSumLayer reduce_sum_layer(in_tensor, out_tensor, in_out_buf, axis, dev_id);

  in_out_buf->init(dev_id);

  int in_size = 1;
  for(auto dim : in_dims) {
    in_size *= dim;
  }
  auto out_dims = out_tensor->get_dims();
  int out_size = 1;
  for(auto dim : out_dims) {
    out_size *= dim;
  }

  float* d_in = in_tensor->get_ptr();
  float* d_out = out_tensor->get_ptr();
  std::unique_ptr<float[]> h_in(new float [in_size]);
  std::unique_ptr<float[]> h_out(new float[out_size]);
  std::unique_ptr<float[]> h_cpu_out(new float[out_size]);
  std::unique_ptr<float[]> h_gpu_dgrad(new float[in_size]);

  // fprop
  for (int i = 0; i < in_size; i++) {
    h_in[i] = simulator.get_num();
  }
  // for(int i = 0; i < in_dims[0]; i++) {
  //   for(int j = 0; j < in_dims[1]; j++) {
  //     for(int k = 0; k < in_dims[2]; k++) {
  //       h_in[i*in_dims[1]*in_dims[2]+j*in_dims[2]+k] = i*in_dims[1]+j;
  //     }
  //   }
  // }

  // std::cout << "axis=" << axis << std::endl;
  // std::cout << "data in:" << std::endl;
  // for(int i = 0; i < in_dims[0]; i++) {
  //   for(int j = 0; j < in_dims[1]; j++) {
  //     for(int k = 0; k < in_dims[2]; k++) {
  //       std::cout << h_in[i*in_dims[1]*in_dims[2]+j*in_dims[2]+k] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }

  cudaMemcpy(d_in, h_in.get(), in_size * sizeof(float), cudaMemcpyHostToDevice);
  reduce_sum_layer.fprop(cudaStreamDefault);
  cudaMemcpy(h_out.get(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);

  // std::cout << "gpu out:" << std::endl;
  // for(int i = 0; i < out_dims[0]; i++) {
  //   for(int j = 0; j < out_dims[1]; j++) {
  //     for(int k = 0; k < out_dims[2]; k++) {
  //       std::cout << h_out[i*out_dims[1]*out_dims[2]+j*out_dims[2]+k] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }

  reduce_sum_cpu(h_in.get(), h_cpu_out.get(), in_dims[0], in_dims[1], in_dims[2], axis);

  // std::cout << "cpu out:" << std::endl;
  // for(int i = 0; i < out_dims[0]; i++) {
  //   for(int j = 0; j < out_dims[1]; j++) {
  //     for(int k = 0; k < out_dims[2]; k++) {
  //       std::cout << h_cpu_out[i*out_dims[1]*out_dims[2]+j*out_dims[2]+k] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }

  ASSERT_TRUE(test::compare_array_approx<float>(h_out.get(), h_cpu_out.get(), out_size, eps));

  // bprop
  for(int i = 0; i < out_size; i++) {
    h_out[i] = simulator.get_num(); // top_grad
  }
  cudaMemcpy(d_out, h_out.get(), out_size * sizeof(float), cudaMemcpyHostToDevice);
  reduce_sum_layer.bprop(cudaStreamDefault); // compute wgrad and dgrad
  cudaMemcpy(h_gpu_dgrad.get(), d_in, in_size * sizeof(float), cudaMemcpyDeviceToHost);

  reduce_sum_dgrad_cpu(h_out.get(), h_in.get(), in_dims[0], in_dims[1], in_dims[2], axis);
  ASSERT_TRUE(test::compare_array_approx<float>(h_in.get(), h_gpu_dgrad.get(), in_size, eps)); // compare dgrad
}

}  // namespace

TEST(reduce_sum_layer, fprop_and_bprop) {
  reduce_sum_test(2, 3, 4, 0);
  reduce_sum_test(2, 3, 4, 1);
  reduce_sum_test(2, 3, 4, 2);
  reduce_sum_test(40960, 39, 1, 1);
}

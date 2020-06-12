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

void reduce_sum_cpu(const float* input, float* output, std::vector<size_t> dims, size_t axis) {
  if (axis == 0) {
    if (dims.size() == 1) {
      for (size_t i = 0; i < dims[0]; i++) {
        output[0] = input[i];
      }
    } else if (dims.size() == 2) {
      for (size_t k = 0; k < dims[1]; k++) {
        output[k] = 0;
        for (size_t i = 0; i < dims[0]; i++) {
          output[k] += input[i * dims[1] + k];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t k = 0; k < dims[2]; k++) {
          output[j * dims[2] + k] = 0;
          for (size_t i = 0; i < dims[0]; i++) {
            output[j * dims[2] + k] += input[i * dims[1] * dims[2] + j * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 1) {
    if (dims.size() == 2) {
      for (size_t i = 0; i < dims[0]; i++) {
        output[i] = 0;
        for (size_t j = 0; j < dims[1]; j++) {
          output[i] += input[i * dims[1] + j];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t i = 0; i < dims[0]; i++) {
        for (size_t k = 0; k < dims[2]; k++) {
          output[i * dims[2] + k] = 0;
          for (size_t j = 0; j < dims[1]; j++) {
            output[i * dims[2] + k] += input[i * dims[1] * dims[2] + j * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 2) {
    for (size_t i = 0; i < dims[0]; i++) {
      for (size_t j = 0; j < dims[1]; j++) {
        output[i * dims[1] + j] = 0;
        for (size_t k = 0; k < dims[2]; k++) {
          output[i * dims[1] + j] += input[i * dims[1] * dims[2] + j * dims[2] + k];
        }
      }
    }
  }
}

void reduce_sum_dgrad_cpu(const float* top_grad, float* dgrad, std::vector<size_t> dims, size_t axis) {
  if (axis == 0) {
    if (dims.size() == 2) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t i = 0; i < dims[0]; i++) {
          dgrad[i * dims[1] + j] = top_grad[j];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t k = 0; k < dims[2]; k++) {
          for (size_t i = 0; i < dims[0]; i++) {
            dgrad[i * dims[1] * dims[2] + j * dims[2] + k] = top_grad[j * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 1) {
    if (dims.size() == 2) {
      for (size_t i = 0; i < dims[0]; i++) {
        for (size_t j = 0; j < dims[1]; j++) {
          dgrad[i * dims[1] + j] = top_grad[i];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t i = 0; i < dims[0]; i++) {
        for (size_t k = 0; k < dims[2]; k++) {
          for (size_t j = 0; j < dims[1]; j++) {
            dgrad[i * dims[1] * dims[2] + j * dims[2] + k] = top_grad[i * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 2) {
    for (size_t i = 0; i < dims[0]; i++) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t k = 0; k < dims[2]; k++) {
          dgrad[i * dims[1] * dims[2] + j * dims[2] + k] = top_grad[i * dims[1] + j];
        }
      }
    }
  }
}

void reduce_sum_test(size_t batch_size, size_t slot_num, size_t embedding_vec_size, size_t axis) {
  size_t dev_id = 0;
  std::shared_ptr<GeneralBuffer<float>> in_out_buf(new GeneralBuffer<float>());

  vector<size_t> in_dims = {batch_size, slot_num, embedding_vec_size};  // 3D
  // vector<size_t> in_dims = {batch_size, slot_num}; // 2D
  TensorFormat_t in_format;
  if (in_dims.size() == 2) {
    in_format = TensorFormat_t::HW;
  } else if (in_dims.size() == 3) {
    in_format = TensorFormat_t::HSW;
  }
  std::shared_ptr<Tensor<float>> in_tensor(new Tensor<float>(in_dims, in_out_buf, in_format));
  std::shared_ptr<Tensor<float>> out_tensor;

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  ReduceSumLayer reduce_sum_layer(in_tensor, out_tensor, in_out_buf, axis, dev_id);

  in_out_buf->init(dev_id);

  size_t in_size = 1;
  for (auto dim : in_dims) {
    in_size *= dim;
  }
  auto out_dims = out_tensor->get_dims();
  size_t out_size = 1;
  for (auto dim : out_dims) {
    out_size *= dim;
  }

  float* d_in = in_tensor->get_ptr();
  float* d_out = out_tensor->get_ptr();
  std::unique_ptr<float[]> h_in(new float[in_size]);
  std::unique_ptr<float[]> h_out(new float[out_size]);
  std::unique_ptr<float[]> h_cpu_out(new float[out_size]);
  std::unique_ptr<float[]> h_gpu_dgrad(new float[in_size]);

  // fprop
  for (size_t i = 0; i < in_size; i++) {
    h_in[i] = simulator.get_num();
  }

  // if(in_dims.size() == 2) {
  //   for(size_t i = 0; i < in_dims[0]; i++) {
  //     for(size_t j = 0; j < in_dims[1]; j++) {
  //       h_in[i*in_dims[1]+j] = i;
  //     }
  //   }
  // }
  // else if(in_dims.size() == 3) {
  //   for(size_t i = 0; i < in_dims[0]; i++) {
  //     for(size_t j = 0; j < in_dims[1]; j++) {
  //       for(size_t k = 0; k < in_dims[2]; k++) {
  //         h_in[i*in_dims[1]*in_dims[2]+j*in_dims[2]+k] = i*in_dims[1]+j;
  //       }
  //     }
  //   }
  // }

  // std::cout << "axis=" << axis << std::endl;
  // std::cout << "data in:" << std::endl;
  // if(in_dims.size() == 2) {
  //   for(size_t i = 0; i < in_dims[0]; i++) {
  //     for(size_t j = 0; j < in_dims[1]; j++) {
  //         std::cout << h_in[i*in_dims[1]+j] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }
  // if(in_dims.size() == 3) {
  //   for(size_t i = 0; i < in_dims[0]; i++) {
  //     for(size_t j = 0; j < in_dims[1]; j++) {
  //       for(size_t k = 0; k < in_dims[2]; k++) {
  //         std::cout << h_in[i*in_dims[1]*in_dims[2]+j*in_dims[2]+k] << " ";
  //       }
  //       std::cout << std::endl;
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  cudaMemcpy(d_in, h_in.get(), in_size * sizeof(float), cudaMemcpyHostToDevice);
  reduce_sum_layer.fprop(cudaStreamDefault);
  cudaMemcpy(h_out.get(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);

  // std::cout << "gpu out:" << std::endl;
  // if(out_dims.size() == 2) {
  //   for(size_t i = 0; i  < out_dims[0]; i++) {
  //     for(size_t j = 0; j < out_dims[1]; j++) {
  //       std::cout << h_out[i*out_dims[1]+j] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }
  // else if(out_dims.size() == 3) {
  //   for(size_t i = 0; i  < out_dims[0]; i++) {
  //     for(size_t j = 0; j < out_dims[1]; j++) {
  //       for(size_t k = 0; k < out_dims[2]; k++) {
  //         std::cout << h_out[i*out_dims[1]*out_dims[2]+j*out_dims[2]+k] << " ";
  //       }
  //       std::cout << std::endl;
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  reduce_sum_cpu(h_in.get(), h_cpu_out.get(), in_dims, axis);

  // std::cout << "cpu out:" << std::endl;
  // if(out_dims.size() == 2) {
  //   for(size_t i = 0; i < out_dims[0]; i++) {
  //     for(size_t j = 0; j < out_dims[1]; j++) {
  //         std::cout << h_cpu_out[i*out_dims[1]+j] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }
  // else if(out_dims.size() == 3) {
  //   for(size_t i = 0; i < out_dims[0]; i++) {
  //     for(size_t j = 0; j < out_dims[1]; j++) {
  //       for(size_t k = 0; k < out_dims[2]; k++) {
  //         std::cout << h_cpu_out[i*out_dims[1]*out_dims[2]+j*out_dims[2]+k] << " ";
  //       }
  //       std::cout << std::endl;
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  ASSERT_TRUE(test::compare_array_approx<float>(h_out.get(), h_cpu_out.get(), out_size, eps));

  // bprop
  for (size_t i = 0; i < out_size; i++) {
    h_out[i] = simulator.get_num();  // top_grad
  }
  cudaMemcpy(d_out, h_out.get(), out_size * sizeof(float), cudaMemcpyHostToDevice);
  reduce_sum_layer.bprop(cudaStreamDefault);  // compute wgrad and dgrad
  cudaMemcpy(h_gpu_dgrad.get(), d_in, in_size * sizeof(float), cudaMemcpyDeviceToHost);

  reduce_sum_dgrad_cpu(h_out.get(), h_in.get(), in_dims, axis);
  ASSERT_TRUE(test::compare_array_approx<float>(h_in.get(), h_gpu_dgrad.get(), in_size,
                                                eps));  // compare dgrad
}

}  // namespace

TEST(reduce_sum_layer, fprop_and_bprop) {
  reduce_sum_test(2, 3, 4, 0);
  reduce_sum_test(2, 3, 4, 1);
  reduce_sum_test(2, 3, 4, 2);
  reduce_sum_test(40960, 39, 1, 1);
}

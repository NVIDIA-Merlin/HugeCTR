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

#include "HugeCTR/include/layers/cast_layer.hpp"

#include <curand.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "cublas_v2.h"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-6;

void cast_test(size_t dim0, size_t dim1) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  vector<size_t> dims = {dim0, dim1};
  Tensor2<float> in_tensor;
  buff->reserve(dims, &in_tensor);
  Tensor2<__half> out_tensor;
  buff->reserve(dims, &out_tensor);

  CastLayer cast_layer(in_tensor, out_tensor, test::get_default_gpu());

  buff->allocate();
  cast_layer.initialize();

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  const int len = dim0 * dim1;
  float* d_in = in_tensor.get_ptr();
  __half* d_out = out_tensor.get_ptr();

  std::unique_ptr<float[]> h_in(new float[len]);
  std::unique_ptr<__half[]> h_out(new __half[len]);

  simulator.fill(h_in.get(), len);
  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), len * sizeof(float), cudaMemcpyHostToDevice));

  // fprop test
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  cast_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  std::unique_ptr<__half[]> h_out_gpu(new __half[len]);
  CK_CUDA_THROW_(cudaMemcpy(h_out_gpu.get(), d_out, len * sizeof(__half), cudaMemcpyDeviceToHost));

  for (int i = 0; i < len; i++) {
    h_out[i] = h_in[i];
  }
  ASSERT_TRUE(test::compare_array_approx<__half>(h_out.get(), h_out_gpu.get(), len, eps));

  // bprop test
  // doing nothing in bprop, no need to test
  cast_layer.bprop();
}

TEST(cast_layer, 32x64) { cast_test(32, 64); }
TEST(cast_layer, 64x128) { cast_test(64, 128); }
TEST(cast_layer, 128x256) { cast_test(128, 256); }
TEST(cast_layer, 256x512) { cast_test(256, 512); }

}  // end namespace

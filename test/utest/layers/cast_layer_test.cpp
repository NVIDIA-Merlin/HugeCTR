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

void cast_cpu(__half* top, const float* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = __float2half(bottom[i]);
  }
}

void cast_cpu(float* top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = __half2float(bottom[i]);
  }
}

template <typename From, typename To>
void cast_test(size_t dim0, size_t dim1) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  vector<size_t> dims = {dim0, dim1};
  Tensor2<From> in_tensor;
  buff->reserve(dims, &in_tensor);
  Tensor2<To> out_tensor;
  buff->reserve(dims, &out_tensor);

  CastLayer<From, To> cast_layer(in_tensor, out_tensor, test::get_default_gpu());

  buff->allocate();
  cast_layer.initialize();

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  const int len = dim0 * dim1;
  From* d_in = in_tensor.get_ptr();
  To* d_out = out_tensor.get_ptr();

  std::unique_ptr<From[]> h_in(new From[len]);
  std::unique_ptr<To[]> h_out(new To[len]);

  simulator.fill(h_in.get(), len);
  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), len * sizeof(From), cudaMemcpyHostToDevice));

  // fprop test
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  cast_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  std::unique_ptr<To[]> h_out_gpu(new To[len]);
  CK_CUDA_THROW_(cudaMemcpy(h_out_gpu.get(), d_out, len * sizeof(To), cudaMemcpyDeviceToHost));

  cast_cpu(h_out.get(), h_in.get(), len);
  ASSERT_TRUE(test::compare_array_approx<To>(h_out.get(), h_out_gpu.get(), len, eps));

  // bprop test
  // doing nothing in bprop, no need to test
  cast_layer.bprop();
}

TEST(cast_layer, fp32_fp16_32x64) { cast_test<float, __half>(32, 64); }
TEST(cast_layer, fp32_fp16_64x128) { cast_test<float, __half>(64, 128); }
TEST(cast_layer, fp32_fp16_128x256) { cast_test<float, __half>(128, 256); }
TEST(cast_layer, fp32_fp16_256x512) { cast_test<float, __half>(256, 512); }
TEST(cast_layer, fp16_fp32_32x64) { cast_test<__half, float>(32, 64); }
TEST(cast_layer, fp16_fp32_64x128) { cast_test<__half, float>(64, 128); }
TEST(cast_layer, fp16_fp32_128x256) { cast_test<__half, float>(128, 256); }
TEST(cast_layer, fp16_fp32_256x512) { cast_test<__half, float>(256, 512); }

}  // end namespace

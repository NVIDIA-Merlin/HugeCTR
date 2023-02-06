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

#include <layers/scale_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-6;

template <typename T>
void upscale_cpu(T* out, T* in, int batchsize, int num_elems, int axis, int factor) {
  int W = num_elems, H = batchsize;

  if (axis == 0) {
    int count = 0;
    for (int i = 0; i < H * W; i++) {
      for (int j = 0; j < factor; j++) {
        out[count] = in[i];
        count++;
      }
    }
  } else {
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < factor; j++) {
        memcpy(&out[int(i * factor * W) + j * W], &in[i * W], W * sizeof(T));
        // for(int k = 0; k < W; k++){
        //  out[i * factor * num_elems + j * num_elems + k] = in[k + i * num_elems];
        //}
      }
    }
  }
}

template <typename T>
void downscale_cpu(T* out, T* in, int batchsize, int num_elems, int axis, int factor) {
  int W = num_elems;
  int H = batchsize;

  if (axis == 0) {
    for (int i = 0; i < H * W; i++) {
      out[i] = in[i * factor];
    }
  } else {
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        out[i * W + j] = in[i * W * factor + j];
      }
    }
  }
}

template <typename T>
void scale_test(size_t batchsize, size_t num_elems, int axis, int factor) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
  std::vector<size_t> dims = {batchsize, num_elems};

  Tensor2<T> in_tensor;
  buf->reserve(dims, &in_tensor);
  Tensor2<T> out_tensor;

  ScaleLayer<T> scale_layer(in_tensor, out_tensor, buf, axis, factor, test::get_default_gpu());

  buf->allocate();
  scale_layer.initialize();

  const size_t len = num_elems * batchsize;
  const size_t top_len = num_elems * batchsize * factor;

  std::unique_ptr<T[]> h_bottom(new T[len]);
  std::unique_ptr<T[]> d2h_bottom(new T[len]);
  std::unique_ptr<T[]> h_top(new T[top_len]);
  std::unique_ptr<T[]> d2h_top(new T[top_len]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_bottom.get(), len);

  HCTR_LIB_THROW(
      cudaMemcpy(in_tensor.get_ptr(), h_bottom.get(), len * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  scale_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(
      cudaMemcpy(d2h_top.get(), out_tensor.get_ptr(), top_len * sizeof(T), cudaMemcpyDeviceToHost));

  upscale_cpu<T>(h_top.get(), h_bottom.get(), batchsize, num_elems, axis, factor);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.get(), h_top.get(), top_len, eps));
  // T *t=d2h_top.get(), *t1 = h_top.get(), *in=h_bottom.get();
  // for(int i=0;i<100;i++)
  //  HCTR_LOG(INFO, WORLD, "%f %f %f\n", t[i], t1[i], in[i]);
  // bprop
  simulator.fill(h_top.get(), len);
  // T *tt = h_top.get();
  // HCTR_LOG(INFO, WORLD, "back cpu %f\n", tt[0]);
  HCTR_LIB_THROW(
      cudaMemcpy(out_tensor.get_ptr(), h_top.get(), top_len * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  scale_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(
      cudaMemcpy(d2h_bottom.get(), in_tensor.get_ptr(), len * sizeof(T), cudaMemcpyDeviceToHost));
  downscale_cpu<T>(h_bottom.get(), h_top.get(), batchsize, num_elems, axis, factor);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_bottom.get(), h_bottom.get(), len, eps));
}

}  // namespace
TEST(scale_layer, fp32_100x100_0) { scale_test<float>(100, 100, 0, 3); }
TEST(scale_layer, fp32_512x1024_0) { scale_test<float>(512, 1024, 0, 20); }
TEST(scale_layer, fp32_2048x4096_0) { scale_test<float>(2048, 4096, 0, 42); }
TEST(scale_layer, fp32_100x100_1) { scale_test<float>(100, 100, 1, 3); }
TEST(scale_layer, fp32_512x1024_1) { scale_test<float>(512, 1024, 1, 17); }
TEST(scale_layer, fp32_2048x4096_1) { scale_test<float>(2048, 4096, 1, 42); }
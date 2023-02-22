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
#include <gtest/gtest.h>

#include <core23/cuda_stream.hpp>
#include <core23/curand_generator.hpp>
#include <core23/data_type_helpers.cuh>
#include <core23/low_level_primitives.hpp>
#include <core23/tensor.hpp>
#include <layers/interaction_layer.hpp>
#include <memory>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

template <typename T>
T get_eps(bool use_tf32 = false);

template <>
float get_eps(bool use_tf32) {
  return (use_tf32 ? 5e-1 : 1e-3);
}

template <>
__half get_eps(bool use_tf32) {
  return __float2half(1);
}

template <typename T>
void interaction_layer_test(int64_t height, int64_t n_emb, int64_t in_width,
                            bool enable_tf32_compute = false) {
  std::vector<core23::Tensor> bottom_tensors;

  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::CUDAStream stream(cudaStreamDefault, 0);

  std::vector<std::vector<T>> h_bottoms;
  core23::Shape mlp_shape = {height, in_width};
  core23::Shape emb_shape = {height, n_emb, in_width};
  for (int ni = 0; ni < 2; ni++) {
    bottom_tensors.emplace_back(tensor_params.shape(ni == 0 ? mlp_shape : emb_shape));
    h_bottoms.emplace_back(bottom_tensors[ni].num_elements(),
                           core23::TypeConverter<T, float>::value(0.0f));
    test::normal_sync_cpu(h_bottoms[ni].data(), h_bottoms[ni].size(), 0.f, 1.f, generator);
  }

  auto& bottom_mlp_tensor = bottom_tensors[0];
  auto& bottom_emb_tensor = bottom_tensors[1];

  auto& h_bottom_mlp = h_bottoms[0];
  auto& h_bottom_emb = h_bottoms[1];

  size_t n_ins = 1 + n_emb;
  size_t out_width = n_ins * in_width;

  std::vector<T> h_concat(height * out_width, core23::TypeConverter<T, float>::value(0.0f));
  auto concat_op = [&](bool fprop) {
    for (size_t ni = 0; ni < n_ins; ni++) {
      for (size_t h = 0; h < height; h++) {
        size_t in_idx_base = (ni == 0) ? h * in_width : h * in_width * n_emb;
        for (size_t w = 0; w < in_width; w++) {
          size_t in_idx = in_idx_base + w;
          size_t out_idx = h * out_width + ni * in_width + w;
          if (fprop) {
            h_concat[out_idx] =
                (ni == 0) ? h_bottom_mlp[in_idx] : h_bottom_emb[(ni - 1) * in_width + in_idx];
          } else {
            if (ni == 0) {
              h_bottom_mlp[in_idx] = h_bottom_mlp[in_idx] + h_concat[out_idx];
            } else {
              h_bottom_emb[in_idx + (ni - 1) * in_width] = h_concat[out_idx];
            }
          }
        }
      }
    }
  };
  core23::Tensor top_tensor;
  InteractionLayer<T> interaction_layer(bottom_mlp_tensor, bottom_emb_tensor, top_tensor,
                                        test::get_default_gpu(), true, enable_tf32_compute);

  interaction_layer.initialize();

  // device fprop
  core23::copy_sync(bottom_mlp_tensor.data(), h_bottom_mlp.data(), bottom_mlp_tensor.num_bytes(),
                    bottom_mlp_tensor.device(), core23::DeviceType::CPU);

  core23::copy_sync(bottom_emb_tensor.data(), h_bottom_emb.data(), bottom_emb_tensor.num_bytes(),
                    bottom_emb_tensor.device(), core23::DeviceType::CPU);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  interaction_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  // host fprop
  concat_op(true);
  // check phase 0: concat

  if (n_ins > 31) {
    auto concat_dev_tensor = interaction_layer.get_intermediate(0);
    std::vector<T> h_concat_dev(height * out_width, core23::TypeConverter<T, float>::value(0.0f));
    core23::copy_sync(h_concat_dev.data(), concat_dev_tensor.data(), concat_dev_tensor.num_bytes(),
                      core23::DeviceType::CPU, concat_dev_tensor.device());
    ASSERT_TRUE(test::compare_array_approx<T>(&h_concat_dev.front(), &h_concat.front(),
                                              h_concat.size(), get_eps<T>(enable_tf32_compute)));
  }

  std::vector<T> h_matmul(height * n_ins * n_ins, core23::TypeConverter<T, float>::value(0.0f));
  for (size_t p = 0; p < height; p++) {
    size_t concat_stride = n_ins * in_width * p;
    size_t mat_stride = n_ins * n_ins * p;
    for (size_t m = 0; m < n_ins; m++) {
      for (size_t n = 0; n < n_ins; n++) {
        float accum = 0.0f;
        for (size_t k = 0; k < in_width; k++) {
          accum += h_concat[concat_stride + m * in_width + k] *
                   h_concat[concat_stride + n * in_width + k];
        }
        h_matmul[mat_stride + m * n_ins + n] = accum;
      }
    }
  }
  // check phase 2: matmul

  if (n_ins > 31) {
    auto matmul_dev_tensor = interaction_layer.get_intermediate(1);
    std::vector<T> h_matmul_dev(height * n_ins * n_ins,
                                core23::TypeConverter<T, float>::value(0.0f));
    core23::copy_sync(h_matmul_dev.data(), matmul_dev_tensor.data(), matmul_dev_tensor.num_bytes(),
                      core23::DeviceType::CPU, matmul_dev_tensor.device());
    ASSERT_TRUE(test::compare_array_approx<T>(&h_matmul_dev.front(), &h_matmul.front(),
                                              h_matmul.size(), get_eps<T>(enable_tf32_compute)));
  }

  size_t out_len = in_width + (n_ins * (n_ins + 1) / 2 - n_ins) + 1;
  std::vector<T> h_ref(height * out_len, 0.0);
  for (size_t p = 0; p < height; p++) {
    size_t cur_idx = 0;
    size_t out_stride = p * out_len;
    size_t mat_stride = p * n_ins * n_ins;
    for (size_t i = 0; i < in_width; i++) {
      h_ref[out_stride + cur_idx++] = h_bottom_mlp[p * in_width + i];
    }
    for (size_t n = 0; n < n_ins; n++) {
      for (size_t m = 0; m < n_ins; m++) {
        if (n > m) {
          // use h_mat_dev
          h_ref[out_stride + cur_idx++] = h_matmul[mat_stride + m * n_ins + n];
        }
      }
    }
  }

  std::vector<T> h_top(top_tensor.num_elements(), core23::TypeConverter<T, float>::value(0.0f));
  core23::copy_sync(h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());
  ASSERT_TRUE(test::compare_array_approx<T>(&h_top.front(), &h_ref.front(), h_top.size(),
                                            get_eps<T>(enable_tf32_compute)));

  /*
   * bprop() test begins:
   */
  // device bprop
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  interaction_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  // host bprop
  for (size_t p = 0; p < height; p++) {
    size_t cur_idx = 0;
    size_t out_stride = p * out_len;
    size_t mat_stride = p * n_ins * n_ins;
    for (size_t i = 0; i < in_width; i++) {
      h_bottom_mlp[p * in_width + i] = h_ref[out_stride + cur_idx++];
    }
    for (size_t n = 0; n < n_ins; n++) {
      for (size_t m = 0; m < n_ins; m++) {
        h_matmul[mat_stride + m * n_ins + n] =
            (n > m) ? h_ref[out_stride + cur_idx++] : core23::TypeConverter<T, float>::value(0.0f);
      }
    }
  }
  // check phase 0, gather and concat
  std::vector<T> h_concat_tmp(h_concat);
  for (size_t p = 0; p < height; p++) {
    size_t mat_stride = n_ins * n_ins * p;
    size_t concat_stride = n_ins * in_width * p;
    for (size_t m = 0; m < n_ins; m++) {
      for (size_t n = 0; n < in_width; n++) {
        float accum = 0.0f;
        for (size_t k = 0; k < n_ins; k++) {
          accum += (h_matmul[mat_stride + m * n_ins + k] + h_matmul[mat_stride + k * n_ins + m]) *
                   h_concat_tmp[concat_stride + k * in_width + n];
        }
        h_concat[concat_stride + m * in_width + n] = 1.0f * accum;
      }
    }
  }
  std::vector<T> h_mat_tmp(h_matmul);

  for (size_t p = 0; p < height; p++) {
    size_t mat_stride = n_ins * n_ins * p;
    for (size_t m = 0; m < n_ins; m++) {
      for (size_t n = 0; n < n_ins; n++) {
        h_mat_tmp[mat_stride + m * n_ins + n] =
            (h_matmul[mat_stride + m * n_ins + n] + h_matmul[mat_stride + n * n_ins + m]);
      }
    }
  }

  if (n_ins > 31) {
    auto concat_tmp_dev_tensor = interaction_layer.get_intermediate(3);
    std::vector<T> h_mat_dev(h_matmul.size(), core23::TypeConverter<T, float>::value(0.0f));
    core23::copy_sync(h_mat_dev.data(), concat_tmp_dev_tensor.data(),
                      concat_tmp_dev_tensor.num_bytes(), core23::DeviceType::CPU,
                      concat_tmp_dev_tensor.device());
    ASSERT_TRUE(test::compare_array_approx<T>(&h_mat_tmp.front(), &h_mat_dev.front(),
                                              h_mat_dev.size(), get_eps<T>(enable_tf32_compute)));
  }

  concat_op(false);

  for (int i = 0; i < 2; i++) {
    auto bottom_tensor = bottom_tensors[i];
    std::vector<T> h_bottom(bottom_tensor.num_elements(),
                            core23::TypeConverter<T, float>::value(0.0f));
    core23::copy_sync(h_bottom.data(), bottom_tensor.data(), bottom_tensor.num_bytes(),
                      core23::DeviceType::CPU, bottom_tensor.device());
    std::vector<T>& h_ref = h_bottoms[i];

    ASSERT_TRUE(test::compare_array_approx<T>(&h_bottom.front(), &h_ref.front(), h_bottom.size(),
                                              get_eps<T>(enable_tf32_compute)));
  }
}

}  // namespace

TEST(interaction_layer, fp32_512x479) { interaction_layer_test<float>(512, 26, 128); }

TEST(interaction_layer, fp32_512x1340) { interaction_layer_test<float>(512, 33, 128); }
TEST(interaction_layer, tf32_512x479) { interaction_layer_test<float>(512, 26, 128, true); }

TEST(interaction_layer, fp16_512x479) {
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  if (major < 7) {
    GTEST_SKIP();
  }
  interaction_layer_test<__half>(512, 26, 128);
}

TEST(interaction_layer, fp16_512x1340) {
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  if (major < 7) {
    GTEST_SKIP();
  }
  interaction_layer_test<__half>(512, 33, 128);
}
TEST(interaction_layer, fp16_512x8643) {
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  if (major < 7) {
    GTEST_SKIP();
  }
  interaction_layer_test<__half>(512, 130, 128);
}

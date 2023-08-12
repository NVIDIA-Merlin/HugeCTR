#include <bits/stdc++.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <unistd.h>

#include <cmath>
#include <hps/dequantize.hpp>
#include <hps/quantize.hpp>
#include <random>
#include <utils.hpp>

using namespace HugeCTR;

namespace {

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<__nv_fp8_e4m3> : std::true_type {};

template <>
struct is_fp8<__nv_fp8_e5m2> : std::true_type {};

// generate the the size of embedding vector for testing
void fill_vec(float* vals, size_t embedding_vec_size, size_t batch_size, float min = -1,
              float max = 1) {
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < embedding_vec_size; ++j) {
      float vect = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
      vals[i * embedding_vec_size + j] = vect;
    }
  }
}

template <typename OutT>
void quantize_cpu(float* input, OutT* output, float* scales, size_t batch_size,
                  size_t emb_vec_size) {
  for (size_t i = 0; i < batch_size; ++i) {
    float max = 0.f;
    for (size_t j = 0; j < emb_vec_size; ++j) {
      max =
          std::abs(input[i * emb_vec_size + j]) > max ? std::abs(input[i * emb_vec_size + j]) : max;
    }
    float scale = 1.0f;
    if constexpr (is_fp8<OutT>::value) {
      scale = (float)std::max(max / 448.f, 1 / (448.f * 512.f));
    } else {
      scale = max != 0.f ? 127.f / max : 1.f;
    }
    for (size_t j = 0; j < emb_vec_size; ++j) {
      if constexpr (is_fp8<OutT>::value) {
        output[i * emb_vec_size + j] = OutT(input[i * emb_vec_size + j] / scale);
      } else {
        output[i * emb_vec_size + j] = std::forward<OutT>(input[i * emb_vec_size + j] * scale);
      }
    }

    scales[i] = scale;
  }
}

template <typename OutT>
void quantize_test(size_t batch_size, size_t emb_vec_size) {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  float* input;
  cudaHostAlloc((void**)&input, batch_size * emb_vec_size * sizeof(float), cudaHostAllocPortable);
  OutT* output;
  cudaHostAlloc((void**)&output, batch_size * emb_vec_size * sizeof(OutT), cudaHostAllocPortable);
  float* scales;
  cudaHostAlloc((void**)&scales, batch_size * sizeof(float), cudaHostAllocPortable);
  fill_vec(input, emb_vec_size, batch_size);
  quantize_cpu<OutT>(input, output, scales, batch_size, emb_vec_size);

  OutT* d_output;
  cudaHostAlloc((void**)&d_output, batch_size * emb_vec_size * sizeof(OutT), cudaHostAllocPortable);
  float* d_scales;
  cudaHostAlloc((void**)&d_scales, batch_size * sizeof(float), cudaHostAllocPortable);
  float* d_input;
  cudaHostAlloc((void**)&d_input, batch_size * emb_vec_size * sizeof(float), cudaHostAllocPortable);

  if constexpr (!is_fp8<OutT>::value) {
    HugeCTR::Quantize<float, OutT> quantize_int8 = new HugeCTR::Quantize<float, OutT>(false, false);
    quantize_int8.quantize(input, d_output, d_scales, batch_size, emb_vec_size, stream);
    for (size_t i = 0; i < batch_size; ++i) {
      EXPECT_LE(abs(scales[i] - d_scales[i]), 1e-4);
      for (size_t j = 0; j < emb_vec_size; ++j) {
        EXPECT_LE(abs(output[i * emb_vec_size + j] - d_output[i * emb_vec_size + j]), 1e-4);
      }
    }

    HugeCTR::Dequantize<OutT, float>* dequantize_gpu = new HugeCTR::Dequantize<OutT, float>();
    dequantize_gpu->dequantize((d_output), d_input, scales, batch_size, emb_vec_size);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < emb_vec_size; ++j) {
        float bias = input[i * emb_vec_size + j] - d_input[i * emb_vec_size + j];
        EXPECT_LE(abs(bias), 1e-2);
      }
    }
  } else {
    std::cout << "FP8 quantization test!" << std::endl;
    __nv_fp8_e4m3* d_output_fp8;
    cudaHostAlloc((void**)&d_output_fp8, batch_size * emb_vec_size * sizeof(__nv_fp8_e4m3),
                  cudaHostAllocPortable);
    HugeCTR::Quantize<float, __nv_fp8_e4m3> quantize_fp8 =
        new HugeCTR::Quantize<float, __nv_fp8_e4m3>(false, false);
    quantize_fp8.quantize(input, d_output_fp8, d_scales, batch_size, emb_vec_size, stream);
    HugeCTR::Dequantize<__nv_fp8_e4m3, float>* dequantize_fp8 =
        new HugeCTR::Dequantize<__nv_fp8_e4m3, float>();
    dequantize_fp8->dequantize(d_output_fp8, d_input, d_scales, batch_size, emb_vec_size);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < emb_vec_size; ++j) {
        float bias = input[i * emb_vec_size + j] - d_input[i * emb_vec_size + j];
        EXPECT_LE(abs(bias), 5 * 1e-2);
      }
    }
    cudaFreeHost(d_output_fp8);
  }

  cudaFreeHost(input);
  cudaFreeHost(output);
  cudaFreeHost(scales);
  cudaFreeHost(d_output);
  cudaFreeHost(d_scales);
  cudaFreeHost(d_input);
}

}  // namespace

TEST(quantize_test, CPU_quantize_fp8_1024_1) { quantize_test<__nv_fp8_e4m3>(1024, 1); }

TEST(quantize_test, CPU_quantize_fp8_1024_32) { quantize_test<__nv_fp8_e4m3>(1024, 32); }

TEST(quantize_test, CPU_quantize_fp8_1024_16) { quantize_test<__nv_fp8_e4m3>(1024, 16); }

TEST(quantize_test, CPU_quantize_fp8_1024_128) { quantize_test<__nv_fp8_e4m3>(1024, 128); }

TEST(quantize_test, CPU_quantize_int8_1024_1) { quantize_test<int8_t>(1024, 1); }

TEST(quantize_test, CPU_quantize_int8_1024_32) { quantize_test<int8_t>(1024, 32); }

TEST(quantize_test, CPU_quantize_int8_1024_16) { quantize_test<int8_t>(1024, 16); }

TEST(quantize_test, CPU_quantize_int8_1024_128) { quantize_test<int8_t>(1024, 128); }

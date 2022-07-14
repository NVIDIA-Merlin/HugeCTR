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

#include <gtest/gtest.h>
#include <test/prims/test_utils.h>

#include <cuda_utils.cuh>
#include <prims/linalg/reduce.cuh>
#include <random/rng.cuh>

#include "reduce.h"

namespace MLCommon {
namespace LinAlg {

template <typename T>
struct ReduceInputs {
  T tolerance;
  int rows, cols;
  bool rowMajor, alongRows;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const ReduceInputs<T> &dims) {
  return os;
}

template <typename T>
void reduceLaunch(T *dots, T **data, int cols, int rows, bool rowMajor, bool alongRows,
                  bool inplace, cudaStream_t stream) {
  reduce(dots, data, cols, rows, (T)0, rowMajor, alongRows, stream, inplace,
         [] __device__(T in, int i) { return in; });
}

template <typename T>
class ReduceTest : public ::testing::TestWithParam<ReduceInputs<T>> {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    reduceTest();
  }

  void reduceTest() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Random::Rng r(params.seed);
    int rows = params.rows, cols = params.cols;
    outlen2d = rows;

    cudaMallocHost((void **)(&h_data_2d), rows * sizeof(T *));
    for (int i = 0; i < rows; i++) {
      allocate(h_data_2d[i], cols);
      r.uniform(h_data_2d[i], cols, T(-1.0), T(1.0), stream);
    }

    cudaMalloc((void **)(&d_data_2d), rows * sizeof(T *));
    cudaMemcpy(d_data_2d, h_data_2d, rows * sizeof(T *), cudaMemcpyHostToDevice);

    allocate(output_act, rows);
    allocate(output_ext, rows);

    cudaEventRecord(start);
    reduceLaunch(output_act, d_data_2d, cols, rows, params.rowMajor, params.alongRows, false,
                 stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Coalesced: " << milliseconds << " miliseconds" << std::endl;

    cudaEventRecord(start);
    naiveReduction2d(output_ext, d_data_2d, cols, rows, params.rowMajor, params.alongRows, stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // std::cout << "Naive: " << milliseconds << " miliseconds" << std::endl;
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(d_data_2d));
    CUDA_CHECK(cudaFreeHost(h_data_2d));
    CUDA_CHECK(cudaFree(output_ext));
    CUDA_CHECK(cudaFree(output_act));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  ReduceInputs<T> params;
  T **d_data_2d;
  T **h_data_2d;
  T *output_ext;
  T *output_act;
  int outlen;
  int outlen2d;
  cudaStream_t stream;
};

const std::vector<ReduceInputs<float>> inputsf = {{0.000002f, 4, 2, true, true, 1234ULL},
                                                  {0.000002f, 4, 2, true, false, 1234ULL},
                                                  {0.000002f, 4, 2, false, true, 1234ULL},
                                                  {0.000002f, 4, 2, false, false, 1234ULL}};

const std::vector<ReduceInputs<double>> inputsd = {{0.000000001, 4, 2, true, true, 1234ULL},
                                                   {0.000000001, 4, 2, true, false, 1234ULL},
                                                   {0.000000001, 4, 2, false, true, 1234ULL},
                                                   {0.000000001, 4, 2, false, false, 1234ULL}};

/*
const std::vector<ReduceInputs<float>> inputsf = {
  {0.000002f, 1024, 32, true, true, 1234ULL},
  {0.000002f, 1024, 64, true, true, 1234ULL},
  {0.000002f, 1024, 128, true, true, 1234ULL},
  {0.000002f, 1024, 256, true, true, 1234ULL},
  {0.000002f, 1024, 32, true, false, 1234ULL},
  {0.000002f, 1024, 64, true, false, 1234ULL},
  {0.000002f, 1024, 128, true, false, 1234ULL},
  {0.000002f, 1024, 256, true, false, 1234ULL},
  {0.000002f, 1024, 32, false, true, 1234ULL},
  {0.000002f, 1024, 64, false, true, 1234ULL},
  {0.000002f, 1024, 128, false, true, 1234ULL},
  {0.000002f, 1024, 256, false, true, 1234ULL},
  {0.000002f, 1024, 32, false, false, 1234ULL},
  {0.000002f, 1024, 64, false, false, 1234ULL},
  {0.000002f, 1024, 128, false, false, 1234ULL},
  {0.000002f, 1024, 256, false, false, 1234ULL},
  {0.000002f, 256, 1024, false, false, 1234ULL}};

const std::vector<ReduceInputs<double>> inputsd = {
  {0.000000001, 1024, 32, true, true, 1234ULL},
  {0.000000001, 1024, 64, true, true, 1234ULL},
  {0.000000001, 1024, 128, true, true, 1234ULL},
  {0.000000001, 1024, 256, true, true, 1234ULL},
  {0.000000001, 1024, 32, true, false, 1234ULL},
  {0.000000001, 1024, 64, true, false, 1234ULL},
  {0.000000001, 1024, 128, true, false, 1234ULL},
  {0.000000001, 1024, 256, true, false, 1234ULL},
  {0.000000001, 1024, 32, false, true, 1234ULL},
  {0.000000001, 1024, 64, false, true, 1234ULL},
  {0.000000001, 1024, 128, false, true, 1234ULL},
  {0.000000001, 1024, 256, false, true, 1234ULL},
  {0.000000001, 1024, 32, false, false, 1234ULL},
  {0.000000001, 1024, 64, false, false, 1234ULL},
  {0.000000001, 1024, 128, false, false, 1234ULL},
  {0.000000001, 1024, 256, false, false, 1234ULL},
  {0.000002f, 256, 1024, false, false, 1234ULL}};
*/

typedef ReduceTest<float> ReduceTestF;
TEST_P(ReduceTestF, Result) {
  ASSERT_TRUE(
      devArrMatch(output_ext, output_act, outlen2d, CompareApprox<float>(params.tolerance)));
}

typedef ReduceTest<double> ReduceTestD;
TEST_P(ReduceTestD, Result) {
  ASSERT_TRUE(
      devArrMatch(output_ext, output_act, outlen2d, CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(ReduceTests, ReduceTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(ReduceTests, ReduceTestD, ::testing::ValuesIn(inputsd));

}  // end namespace LinAlg
}  // end namespace MLCommon

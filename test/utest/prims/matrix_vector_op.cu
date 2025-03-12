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
#include <utest/prims/matrix_vector_op.h>

#include <linalg/matrix_vector_op.cuh>
#include <linalg/unary_op.cuh>
#include <random/rng.cuh>

namespace MLCommon {
namespace LinAlg {
//! we intentionally avoid using third_party/cuml/cpp/test/prims/test_utils.h
//! because its misuse of shared_ptr of array. We define used util here
// CompareApprox
template <typename T>
struct CompareApprox {
  CompareApprox(T eps_) : eps(eps_) {}
  bool operator()(const T &a, const T &b) const {
    T diff = abs(a - b);
    T m = std::max(abs(a), abs(b));
    T ratio = diff >= eps ? diff / m : diff;

    return (ratio <= eps);
  }

 private:
  T eps;
};
// match
template <typename T, typename L>
::testing::AssertionResult devArrMatch(const T *expected, const T *actual, size_t size,
                                       L eq_compare, cudaStream_t stream = 0) {
  std::shared_ptr<T[]> exp_h(new T[size]);
  std::shared_ptr<T[]> act_h(new T[size]);
  updateHost<T>(exp_h.get(), expected, size, stream);
  updateHost<T>(act_h.get(), actual, size, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (size_t i(0); i < size; ++i) {
    auto exp = exp_h.get()[i];
    auto act = act_h.get()[i];
    if (!eq_compare(exp, act)) {
      return ::testing::AssertionFailure()
             << "actual=" << act << " != expected=" << exp << " @" << i;
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename T, typename IdxType = int>
struct MatVecOpInputs {
  T tolerance;
  IdxType rows, cols;
  bool rowMajor, bcastAlongRows, useTwoVectors;
  unsigned long long int seed;
};

template <typename T, typename IdxType>
::std::ostream &operator<<(::std::ostream &os, const MatVecOpInputs<T, IdxType> &dims) {
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T, typename IdxType>
void matrixVectorOpLaunch1(T *out, const T *in, const T *vec1, IdxType D, IdxType N, bool rowMajor,
                           bool bcastAlongRows, cudaStream_t stream) {
  matrixVectorOp(
      out, in, vec1, D, N, rowMajor, bcastAlongRows,
      [] __device__(T a, T b) {
        T in = a + b;
        return (in < 0) ? 0 : in;
      },
      stream);
}

template <typename T, typename IdxType>
void matrixVectorOpLaunch2(T *out, const T *in, const T *vec1, IdxType D, IdxType N, bool rowMajor,
                           bool bcastAlongRows, cudaStream_t stream) {
  matrixVectorOp(
      out, in, vec1, D, N, rowMajor, bcastAlongRows, [] __device__(T a, T b) { return a + b; },
      stream);

  unaryOp(
      out, out, D * N, [] __device__(T in) { return (in < 0) ? 0 : in; }, stream);
}

template <typename T, typename IdxType>
class MatVecOpTest : public ::testing::TestWithParam<MatVecOpInputs<T, IdxType>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MatVecOpInputs<T, IdxType>>::GetParam();
    Random::Rng r(params.seed);
    IdxType N = params.rows, D = params.cols;
    IdxType len = N * D;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(in, len);
    allocate(out_ref, len);
    allocate(out, len);
    IdxType vecLen = params.bcastAlongRows ? D : N;
    allocate(vec1, vecLen);
    allocate(vec2, vecLen);
    r.uniform(in, len, (T)-1.0, (T)1.0, stream);
    r.uniform(vec1, vecLen, (T)-1.0, (T)1.0, stream);
    r.uniform(vec2, vecLen, (T)-1.0, (T)1.0, stream);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matrixVectorOpLaunch1(out_ref, in, vec1, D, N, params.rowMajor, params.bcastAlongRows, stream);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    // HCTR_LOG(INFO, WORLD, "Fused: %f\n", milliseconds);

    CUDA_CHECK(cudaEventRecord(start));
    matrixVectorOpLaunch2(out, in, vec1, D, N, params.rowMajor, params.bcastAlongRows, stream);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    // HCTR_LOG(INFO, WORLD, "Normal: %f\n", milliseconds);

    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(vec1));
    CUDA_CHECK(cudaFree(vec2));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(in));
  }

 protected:
  MatVecOpInputs<T, IdxType> params;
  T *in, *out, *out_ref, *vec1, *vec2;
};

const std::vector<MatVecOpInputs<float, int>> inputsf_i32 = {
    {0.00001f, 10024, 32, true, true, false, 1234ULL},
    {0.00001f, 10024, 64, true, true, false, 1234ULL},
    {0.00001f, 10024, 32, true, false, false, 1234ULL},
    {0.00001f, 10024, 64, true, false, false, 1234ULL},
    {0.00001f, 10024, 32, false, true, false, 1234ULL},
    {0.00001f, 10024, 64, false, true, false, 1234ULL},
    {0.00001f, 10024, 32, false, false, false, 1234ULL},
    {0.00001f, 10024, 64, false, false, false, 1234ULL},
    {0.00001f, 10024, 32, true, true, true, 1234ULL},
    {0.00001f, 10024, 64, true, true, true, 1234ULL},
    {0.00001f, 10024, 32, true, false, true, 1234ULL},
    {0.00001f, 10024, 64, true, false, true, 1234ULL},
    {0.00001f, 10024, 32, false, true, true, 1234ULL},
    {0.00001f, 10024, 64, false, true, true, 1234ULL},
    {0.00001f, 10024, 32, false, false, true, 1234ULL},
    {0.00001f, 10024, 64, false, false, true, 1234ULL}};

typedef MatVecOpTest<float, int> MatVecOpTestF_i32;
TEST_P(MatVecOpTestF_i32, Result) {
  ASSERT_TRUE(
      devArrMatch(out_ref, out, params.rows * params.cols, CompareApprox<float>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MatVecOpTests, MatVecOpTestF_i32, ::testing::ValuesIn(inputsf_i32));

const std::vector<MatVecOpInputs<float, size_t>> inputsf_i64 = {
    {0.00001f, 2500, 250, false, false, false, 1234ULL},
    {0.00001f, 2500, 250, false, false, true, 1234ULL}};

typedef MatVecOpTest<float, size_t> MatVecOpTestF_i64;
TEST_P(MatVecOpTestF_i64, Result) {
  ASSERT_TRUE(
      devArrMatch(out_ref, out, params.rows * params.cols, CompareApprox<float>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MatVecOpTests, MatVecOpTestF_i64, ::testing::ValuesIn(inputsf_i64));

const std::vector<MatVecOpInputs<double, int>> inputsd_i32 = {
    {0.0000001, 10024, 32, true, true, false, 1234ULL},
    {0.0000001, 10024, 64, true, true, false, 1234ULL},
    {0.0000001, 10024, 32, true, false, false, 1234ULL},
    {0.0000001, 10024, 64, true, false, false, 1234ULL},
    {0.0000001, 10024, 32, false, true, false, 1234ULL},
    {0.0000001, 10024, 64, false, true, false, 1234ULL},
    {0.0000001, 10024, 32, false, false, false, 1234ULL},
    {0.0000001, 10024, 64, false, false, false, 1234ULL},
    {0.0000001, 10024, 32, true, true, true, 1234ULL},
    {0.0000001, 10024, 64, true, true, true, 1234ULL},
    {0.0000001, 10024, 32, true, false, true, 1234ULL},
    {0.0000001, 10024, 64, true, false, true, 1234ULL},
    {0.0000001, 10024, 32, false, true, true, 1234ULL},
    {0.0000001, 10024, 64, false, true, true, 1234ULL},
    {0.0000001, 10024, 32, false, false, true, 1234ULL},
    {0.0000001, 10024, 64, false, false, true, 1234ULL}};

typedef MatVecOpTest<double, int> MatVecOpTestD_i32;
TEST_P(MatVecOpTestD_i32, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.rows * params.cols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MatVecOpTests, MatVecOpTestD_i32, ::testing::ValuesIn(inputsd_i32));

const std::vector<MatVecOpInputs<double, size_t>> inputsd_i64 = {
    {0.0000001, 2500, 250, false, false, false, 1234ULL},
    {0.0000001, 2500, 250, false, false, true, 1234ULL}};

typedef MatVecOpTest<double, size_t> MatVecOpTestD_i64;
TEST_P(MatVecOpTestD_i64, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.rows * params.cols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MatVecOpTests, MatVecOpTestD_i64, ::testing::ValuesIn(inputsd_i64));

}  // end namespace LinAlg
}  // end namespace MLCommon

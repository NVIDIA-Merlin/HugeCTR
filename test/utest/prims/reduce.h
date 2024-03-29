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
#pragma once

#include <cublas_v2.h>
#include <linalg/cublas_wrappers.h>
#include <thrust/device_vector.h>

#include <cuda_utils.cuh>
#include <linalg/unary_op.cuh>

namespace MLCommon {
namespace LinAlg {

template <typename Type>
__global__ void naiveCoalescedReductionKernel(Type *dots, const Type *data, int D, int N) {
  Type acc = (Type)0;
  int rowStart = threadIdx.x + blockIdx.x * blockDim.x;
  if (rowStart < N) {
    for (int i = 0; i < D; ++i) {
      acc += data[rowStart * D + i] * data[rowStart * D + i];
    }
    dots[rowStart] = 2 * acc;
  }
}

template <typename Type>
__global__ void naiveCoalescedReductionKernel2d(Type *dots, Type **data, int D, int N,
                                                bool rowMajor) {
  Type acc = (Type)0;
  int start = threadIdx.x + blockIdx.x * blockDim.x;
  if (rowMajor) {
    if (start < N) {
      for (int i = 0; i < D; ++i) {
        acc += data[start][i];
      }
      dots[start] = acc;
    }
  } else {
    if (start < D) {
      for (int i = 0; i < N; ++i) {
        acc += data[i][start];
      }
      dots[start] = acc;
    }
  }
}

template <typename Type>
__global__ void naiveStridedReductionKernel2d(Type *dots, Type **data, int D, int N,
                                              bool rowMajor) {
  Type acc = (Type)0;
  int start = threadIdx.x + blockIdx.x * blockDim.x;
  if (rowMajor) {
    if (start < D) {
      for (int i = 0; i < N; ++i) {
        acc += data[i][start];
      }
      dots[start] = acc;
    }
  } else {
    if (start < N) {
      for (int i = 0; i < D; ++i) {
        acc += data[start][i];
      }
      dots[start] = acc;
    }
  }
}

template <typename Type>
void naiveCoalescedReduction(Type *dots, const Type *data, int D, int N, cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = ceildiv(N, TPB);
  naiveCoalescedReductionKernel<Type><<<nblks, TPB, 0, stream>>>(dots, data, D, N);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
void naiveCoalescedReduction2d(Type *dots, Type **data, int D, int N, bool rowMajor,
                               cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = 0;
  if (rowMajor)
    nblks = ceildiv(N, TPB);
  else
    nblks = ceildiv(D, TPB);

  naiveCoalescedReductionKernel2d<Type><<<nblks, TPB, 0, stream>>>(dots, data, D, N, rowMajor);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
void naiveStridedReduction2d(Type *dots, Type **data, int D, int N, bool rowMajor,
                             cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = 0;
  if (rowMajor)
    nblks = ceildiv(D, TPB);
  else
    nblks = ceildiv(N, TPB);

  naiveStridedReductionKernel2d<Type><<<nblks, TPB, 0, stream>>>(dots, data, D, N, rowMajor);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
void unaryAndGemv(Type *dots, const Type *data, int D, int N, cudaStream_t stream) {
  // computes a MLCommon unary op on data (squares it), then computes Ax
  //(A input matrix and x column vector) to sum columns
  thrust::device_vector<Type> sq(D * N);
  unaryOp(
      thrust::raw_pointer_cast(sq.data()), data, D * N, [] __device__(Type v) { return v * v; },
      stream);
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  thrust::device_vector<Type> ones(N, 1);  // column vector [1...1]
  Type alpha = 1, beta = 0;
  CUBLAS_CHECK(cublasgemv(handle, CUBLAS_OP_N, D, N, &alpha, thrust::raw_pointer_cast(sq.data()), D,
                          thrust::raw_pointer_cast(ones.data()), 1, &beta, dots, 1, stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUBLAS_CHECK(cublasDestroy(handle));
}

template <typename Type>
void naiveReduction(Type *dots, const Type *data, int D, int N, bool rowMajor, bool alongRows,
                    cudaStream_t stream) {
  if (rowMajor && alongRows) {
    naiveCoalescedReduction(dots, data, D, N, stream);
  } else if (rowMajor && !alongRows) {
    unaryAndGemv(dots, data, D, N, stream);
  } else if (!rowMajor && alongRows) {
    unaryAndGemv(dots, data, N, D, stream);
  } else {
    naiveCoalescedReduction(dots, data, N, D, stream);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Type>
void naiveReduction2d(Type *dots, Type **data, int D, int N, bool rowMajor, bool alongRows,
                      cudaStream_t stream) {
  if ((rowMajor && alongRows) || (!rowMajor && !alongRows)) {
    naiveCoalescedReduction2d(dots, data, D, N, rowMajor, stream);
  } else {
    naiveStridedReduction2d(dots, data, D, N, rowMajor, stream);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // end namespace LinAlg
}  // end namespace MLCommon

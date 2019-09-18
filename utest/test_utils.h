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


#pragma once

#include "gtest/gtest.h"
#include <memory>
#include <stdlib.h>

namespace HugeCTR {
  
namespace test {

template <typename T>
T abs(const T& val) {
  return val > T(0)? val : -val;
}

template <typename T>
::testing::AssertionResult compare_array_approx(const T* h_out, const T* h_exp,
                                                int len, T eps) {
  for (int i = 0; i < len; ++i) {
    auto output = h_out[i];
    auto expected = h_exp[i];
    T diff = abs(output - expected);
    if(diff > eps)  {
      return ::testing::AssertionFailure()
          << "output: " << output << " != expected: " << expected << " at idx " << i;
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename T>
::testing::AssertionResult compare_array_approx(const T* h_out, const T expected,
                                                int len, T eps) {
  for (int i = 0; i < len; ++i) {
    auto output = h_out[i];
    T diff = abs(output - expected);
    if(diff > eps)  {
      return ::testing::AssertionFailure()
          << "output: " << output << " != expected: " << expected << " at idx " << i;
    }
  }
  return ::testing::AssertionSuccess();
}

__forceinline__ bool cpu_gpu_cmp(float *cpu_p, float *gpu_p, int len)
{
  float *gpu_tmp = (float*)malloc(sizeof(float) * len);
  cudaMemcpy(gpu_tmp, gpu_p, sizeof(float) * len, cudaMemcpyDeviceToHost);
  bool flag = true;
  for(int i = 0; i < len; ++i){
    if(fabs(gpu_tmp[i] - cpu_p[i]) >= 1e-5){
      printf("gpu_tmp(%f) - cpu_p(%f) >= 1e-5 when i = %d\n",gpu_tmp[i], cpu_p[i], i);
      flag = false;
      break;
    }
  }
  free(gpu_tmp);
  return flag;
}


__forceinline__ void mpi_init(){
#ifdef ENABLE_MPI
  int flag = 0;
  MPI_Initialized( &flag );
  if(!flag){
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }
#endif 
}


 
} // end namespace test

} // end namespace HugeCTRTest

#
# Copyright (c) 2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.20)
set(DB_LIB_PATHS "/usr/local/lib" CACHE PATH "Paths to Hiredis/rocksdb lib")
file(GLOB hps_benchmark_src
  metrics.cpp
)

if(ENABLE_HDFS)
  target_link_libraries(huge_ctr_inference PUBLIC hdfs)
endif()

if(ENABLE_S3)
  target_link_libraries(
    huge_ctr_inference
    PUBLIC
      ${DB_LIB_PATHS}/libaws-cpp-sdk-core.so ${DB_LIB_PATHS}/libaws-cpp-sdk-s3.so # from Hugectr
  )
endif()

add_executable(hps_profiler ${hps_benchmark_src})
target_compile_features(hps_profiler PUBLIC cxx_std_17)
target_link_libraries(hps_profiler PUBLIC hugectr_core23 huge_ctr_hps)
target_link_libraries(hps_profiler PUBLIC ${CUDART_LIB} gtest gtest_main stdc++fs)
target_link_libraries(hps_profiler PUBLIC CUDA::cuda_driver)

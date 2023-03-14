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

cmake_minimum_required(VERSION 3.17)

file(GLOB async_reader_src 
  data_reader_async_adapter_test.cpp
  data_reader_async_test.cpp
)
if (DISABLE_CUDF) 
  file(GLOB data_reader_test_src
    data_reader_test.cpp
    data_reader_raw_test.cpp
  )
else()
  file(GLOB data_reader_test_src
    data_reader_test.cpp
    data_reader_raw_test.cpp
    data_reader_parquet_test.cpp
  )
endif()

add_executable(data_reader_test ${data_reader_test_src})
add_executable(async_reader ${async_reader_src})
add_executable(multi_hot_async_data_reader_test multi_hot_async_data_reader_test.cpp)
add_executable(batch_locations_test batch_locations_test.cpp)
add_executable(v2_async_reader_test data_reader_v2_async_test.cpp)
add_executable(benchmark_async_reader data_reader_benchmark.cu)
target_compile_features(data_reader_test PUBLIC cxx_std_17)
target_link_libraries(async_reader PUBLIC huge_ctr_static gtest gtest_main)
target_link_libraries(async_reader PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)
target_link_libraries(data_reader_test PUBLIC huge_ctr_static gtest gtest_main)
target_link_libraries(data_reader_test PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)
target_link_libraries(v2_async_reader_test PUBLIC huge_ctr_static gtest gtest_main)
target_link_libraries(v2_async_reader_test PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)
target_link_libraries(multi_hot_async_data_reader_test PUBLIC huge_ctr_static gtest gtest_main /usr/local/cuda/lib64/stubs/libcuda.so)
target_link_libraries(benchmark_async_reader PUBLIC huge_ctr_static gtest gtest_main)
target_link_libraries(benchmark_async_reader PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)
set_target_properties(benchmark_async_reader PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(batch_locations_test PUBLIC huge_ctr_static gtest gtest_main /usr/local/cuda/lib64/stubs/libcuda.so)

add_executable(split_test split_batch_test.cpp)
target_link_libraries(split_test PUBLIC huge_ctr_static gtest gtest_main /usr/local/cuda/lib64/stubs/libcuda.so)
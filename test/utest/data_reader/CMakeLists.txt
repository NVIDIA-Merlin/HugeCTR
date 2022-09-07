# 
# Copyright (c) 2021, NVIDIA CORPORATION.
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
target_compile_features(data_reader_test PUBLIC cxx_std_17)
target_link_libraries(async_reader PUBLIC huge_ctr_static gtest gtest_main)
target_link_libraries(async_reader PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)
target_link_libraries(data_reader_test PUBLIC huge_ctr_static gtest gtest_main)
target_link_libraries(data_reader_test PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)


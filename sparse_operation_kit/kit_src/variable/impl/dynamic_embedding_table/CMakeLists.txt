#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
PROJECT(ORION LANGUAGES CXX CUDA)

FIND_PACKAGE(CUDAToolkit REQUIRED)

SET(CMAKE_CUDA_ARCHITECTURES 70;80)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -Wall --expt-relaxed-constexpr")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

INCLUDE_DIRECTORIES(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/cuCollections/include
    ${PROJECT_SOURCE_DIR}/cudf
    ${PROJECT_SOURCE_DIR}/googletest/googletest/include
)

ADD_SUBDIRECTORY(googletest)

FILE(GLOB dynamic_embedding_table_cu_files dynamic_embedding_table.cu)
FILE(GLOB hash_table_cu_files hash_table.cu)

get_property(SOK_CXX_STANDARD_FLAG GLOBAL PROPERTY SOK_CXX_STANDARD_PROPERTY)
get_property(SOK_CUDA_STANDARD_FLAG GLOBAL PROPERTY SOK_CUDA_STANDARD_PROPERTY)

ADD_EXECUTABLE(unit_test ${dynamic_embedding_table_cu_files} test.cu)
TARGET_LINK_LIBRARIES(unit_test CUDA::curand gtest_main)
target_compile_features(unit_test PRIVATE ${SOK_CXX_STANDARD_FLAG} ${SOK_CXX_STANDARD_FLAG})


ADD_EXECUTABLE(benchmark_test ${dynamic_embedding_table_cu_files} benchmark.cu)
TARGET_LINK_LIBRARIES(benchmark_test CUDA::curand gtest_main)
target_compile_features(benchmark_test PRIVATE ${SOK_CXX_STANDARD_FLAG} ${SOK_CXX_STANDARD_FLAG})

ADD_EXECUTABLE(hash_table_benchmark_test ${hash_table_cu_files} hash_table_benchmark.cu)
TARGET_LINK_LIBRARIES(hash_table_benchmark_test CUDA::curand gtest_main)
target_compile_features(hash_table_benchmark_test PRIVATE ${SOK_CXX_STANDARD_FLAG} ${SOK_CXX_STANDARD_FLAG})

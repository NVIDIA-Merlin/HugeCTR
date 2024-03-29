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
find_package(CUDAToolkit)

file(GLOB_RECURSE embedding_test_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)

add_executable(embedding_test ${embedding_test_src})
target_compile_features(embedding_test PUBLIC cxx_std_17)
target_link_libraries(embedding_test PUBLIC CUDA::nvml huge_ctr_shared embedding gtest gtest_main stdc++fs)
target_link_libraries(embedding_test PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)

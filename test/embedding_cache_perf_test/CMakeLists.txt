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
file(GLOB ec_perf_test_src
    ec_perf_test.cu
    key_generator.cuh
)

add_executable(ec_perf_test ${ec_perf_test_src})
set_target_properties(ec_perf_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_features(ec_perf_test PUBLIC cxx_std_17)
target_link_libraries(ec_perf_test PUBLIC huge_ctr_static gtest gtest_main)
target_link_libraries(ec_perf_test PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)

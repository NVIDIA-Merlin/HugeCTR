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
file(GLOB prims_test_src
  matrix_vector_op.cu
  #reduce.cu
)

add_executable(prims_test ${prims_test_src})
set_target_properties(prims_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(prims_test PROPERTIES CUDA_ARCHITECTURES OFF)
target_compile_features(prims_test PUBLIC cxx_std_17)
target_link_libraries(prims_test PUBLIC huge_ctr_static gtest gtest_main)

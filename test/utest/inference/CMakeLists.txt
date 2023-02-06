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

file(GLOB inference_test_src
  embedding_feature_combiner_test.cpp
  preallocated_buffer2_test.cpp
  session_inference_test.cpp
  cpu_inference_test.cpp
  cpu_multicross_layer_test.cpp 
)

add_executable(inference_test ${inference_test_src})
target_compile_features(inference_test PUBLIC cxx_std_17)
target_link_libraries(inference_test PUBLIC cpu_inference_shared gtest gtest_main stdc++fs)
set_target_properties(inference_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(inference_test PROPERTIES CUDA_ARCHITECTURES OFF)
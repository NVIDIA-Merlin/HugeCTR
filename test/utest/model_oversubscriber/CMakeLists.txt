# 
# Copyright (c) 2020, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.8)
file(GLOB model_oversubscriber_test_src
  parameter_server_test.cu
  model_oversubscriber_test.cpp
)

add_executable(model_oversubscriber_test ${model_oversubscriber_test_src})
target_compile_features(model_oversubscriber_test PUBLIC cxx_std_17)
target_link_libraries(model_oversubscriber_test PUBLIC huge_ctr_static gtest gtest_main stdc++fs)
set_target_properties(model_oversubscriber_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(model_oversubscriber_test PROPERTIES CUDA_ARCHITECTURES OFF)

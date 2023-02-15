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
file(GLOB loss_test_src
  loss_test.cpp
  multi_cross_entropy_loss_test.cpp
)

add_executable(loss_test ${loss_test_src})
target_compile_features(loss_test PUBLIC cxx_std_17)
target_link_libraries(loss_test PUBLIC huge_ctr_static gtest gtest_main)
target_link_libraries(loss_test PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)

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
file(GLOB layers_test_src
  batch_norm_layer_test.cpp
  concat_layer_test.cpp
  dropout_layer_test.cpp
  elu_layer_test.cpp
  fully_connected_layer_test.cpp
  fully_connected_layer_half_test.cpp
  fused_fully_connected_layer_test.cpp
  interaction_layer_test.cpp
  relu_layer_test.cpp
  relu_layer_half_test.cpp
  slice_layer_test.cpp
  reshape_layer_test.cpp
  reshape_layer_test_sel.cpp
  fm_order2_layer_test.cpp
  multiply_layer_test.cpp
  multi_cross_layer_test.cpp
  add_layer_test.cpp
  reduce_sum_layer_test.cpp
)

add_executable(layers_test ${layers_test_src})
target_compile_features(layers_test PUBLIC cxx_std_14)
target_link_libraries(layers_test PUBLIC huge_ctr_static gtest gtest_main)

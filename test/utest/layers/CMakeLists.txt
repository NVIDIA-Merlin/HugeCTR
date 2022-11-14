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
file(GLOB layers_test_src
  trainable_layer_test.cpp
  dropout_layer_test.cpp
  batch_norm_layer_test.cpp
  cast_layer_test.cpp
  concat_layer_test.cpp
  elu_layer_test.cpp
  fully_connected_layer_test.cpp
  fully_connected_layer_half_test.cpp
  fused_fully_connected_layer_test.cpp
  fused_relu_bias_fully_connected_layer_test.cpp
  gru_layer_test.cpp
  matrix_multiply_test.cpp
  interaction_layer_test.cpp
  relu_layer_test.cpp
  slice_layer_test.cpp
  reshape_layer_test.cpp
  reshape_layer_test_sel.cpp
  prelu_dice_layer_test.cpp
  fm_order2_layer_test.cpp
  weight_multiply_layer_test.cpp
  multi_cross_layer_test.cpp
  add_layer_test.cpp
  reduce_sum_layer_test.cpp
  softmax_layer_test.cpp
  scale_layer_test.cpp
  fused_reshape_concat_general_layer_test.cpp
  fused_reshape_concat_layer_test.cpp
  sub_layer_test.cpp
  gather_layer_test.cpp
  reduce_mean_layer_test.cpp
  sigmoid_layer_test.cpp
  elementwise_multiply_layer_test.cpp
  group_dense_layer_test.cpp
  mlp_test.cpp
  concat_3d_layer_test.cpp
  multi_head_attention_layer_test.cpp
  layer_norm_layer_test.cpp
  sequence_mask_layer_test.cpp
  masked_softmax_layer_test.cpp
)

add_executable(layers_test ${layers_test_src})
target_compile_features(layers_test PUBLIC cxx_std_17)
target_link_libraries(layers_test PUBLIC huge_ctr_static gtest gtest_main)
target_link_libraries(layers_test PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)

add_custom_command(
  OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python_concat.py"
  COMMAND ${CMAKE_COMMAND} -E copy
  "${CMAKE_CURRENT_SOURCE_DIR}/python_concat.py"
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python_concat.py"
)

add_custom_target(copy_python_auc DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python_concat.py")
add_dependencies(layers_test copy_python_auc)

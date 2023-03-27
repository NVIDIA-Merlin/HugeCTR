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
file(GLOB dense_layer_test_src
  dense_add_layer_test.cpp
  dense_fm_order2_layer_test.cpp
  dense_relu_layer_test.cpp
  dense_reduce_sum_layer_test.cpp
  dense_elu_layer_test.cpp
  dense_loss_test.cpp
  dense_sigmoid_layer_test.cpp
  dense_slice_layer_test.cpp
  dense_dropout_layer_test.cpp
  dense_interaction_layer_test.cpp
  dense_concat_layer_test.cpp
  dense_reshape_layer_test.cpp
  dense_sequence_mask_layer_test.cpp
)

add_executable(dense_layer_test ${dense_layer_test_src})
target_compile_features(dense_layer_test PUBLIC cxx_std_17)
# TODO: remove rocksdb redis++ rdkafka once the pybind dependency is resolved
target_link_libraries(dense_layer_test PUBLIC hugectr_core23 huge_ctr_static rocksdb redis++ rdkafka gtest gtest_main)
target_link_libraries(dense_layer_test PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)
set_target_properties(dense_layer_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(dense_layer_test PROPERTIES CUDA_ARCHITECTURES OFF)

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
file(GLOB_RECURSE layers_test_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)

add_executable(core23_layers_test ${layers_test_src})
target_compile_features(core23_layers_test PUBLIC cxx_std_17)
target_link_libraries(core23_layers_test PUBLIC huge_ctr_shared gtest gtest_main)
target_link_libraries(core23_layers_test PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)

add_custom_command(
  OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python_concat.py"
  COMMAND ${CMAKE_COMMAND} -E copy
  "${CMAKE_CURRENT_SOURCE_DIR}/python_concat.py"
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python_concat.py"
)

add_custom_target(copy_python_auc DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python_concat.py")
add_dependencies(core23_layers_test copy_python_auc)

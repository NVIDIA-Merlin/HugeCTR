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

cmake_minimum_required(VERSION 3.20)

file(GLOB_RECURSE core23_dev_test_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)

add_executable(core23_dev_test ${core23_dev_test_src})
target_compile_features(core23_dev_test PUBLIC cxx_std_17)
target_link_libraries(core23_dev_test PUBLIC hugectr_core23 huge_ctr_shared gtest gtest_main)

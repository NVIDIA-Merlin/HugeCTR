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

cmake_minimum_required(VERSION 3.20)

file(GLOB embedding_collection_test_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)

add_executable(embedding_collection_test ${embedding_collection_test_src})
target_compile_features(embedding_collection_test PUBLIC cxx_std_17)
target_link_libraries(embedding_collection_test PUBLIC gtest gtest_main stdc++fs embedding huge_ctr_shared)
target_link_libraries(embedding_collection_test PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)

add_executable(embedding_table_test_optimizer test_embedding_table_optimizer.cpp)
target_compile_features(embedding_table_test_optimizer PUBLIC cxx_std_17)
target_link_libraries(embedding_table_test_optimizer PUBLIC gtest gtest_main stdc++fs embedding huge_ctr_shared)
target_link_libraries(embedding_table_test_optimizer PUBLIC /usr/local/cuda/lib64/stubs/libcuda.so)

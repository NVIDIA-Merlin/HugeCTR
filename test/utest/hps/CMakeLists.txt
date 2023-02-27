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

file(GLOB embedding_cache_test_src
  embedding_cache_test.cpp
)

file(GLOB embedding_cache_update_test_src
  embedding_cache_update_test.cpp
)

file(GLOB parameter_server_test_src
  parameter_server_test.cpp
)

file(GLOB db_backend_test_src
  db_backend_test.cpp
)

add_executable(embedding_cache_test ${embedding_cache_test_src})
target_compile_features(embedding_cache_test PUBLIC cxx_std_17)
target_link_libraries(embedding_cache_test PUBLIC hugectr_core23 huge_ctr_hps cudart gtest gtest_main stdc++fs)

add_executable(embedding_cache_update_test ${embedding_cache_update_test_src})
target_compile_features(embedding_cache_update_test PUBLIC cxx_std_17)
target_link_libraries(embedding_cache_update_test PUBLIC hugectr_core23 huge_ctr_hps cudart gtest gtest_main stdc++fs)

add_executable(parameter_server_test ${parameter_server_test_src})
target_compile_features(parameter_server_test PUBLIC cxx_std_17)
target_link_libraries(parameter_server_test PUBLIC hugectr_core23 huge_ctr_hps cudart gtest gtest_main stdc++fs)

add_executable(db_backend_test ${db_backend_test_src})
target_compile_features(db_backend_test PUBLIC cxx_std_17)
target_link_libraries(db_backend_test PUBLIC hugectr_core23 huge_ctr_hps cudart gtest gtest_main stdc++fs)

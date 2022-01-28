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
set(DB_LIB_PATHS "/usr/local/lib" CACHE PATH "Paths to Hiredis/RocksDB lib")
file (GLOB hdfs_backend_test_src
  hdfs_backend_test.cpp
)

add_executable(hdfs_backend_test ${hdfs_backend_test_src})
target_compile_features(hdfs_backend_test PUBLIC cxx_std_17)
target_link_libraries(hdfs_backend_test PUBLIC huge_ctr_shared gtest gtest_main stdc++fs)
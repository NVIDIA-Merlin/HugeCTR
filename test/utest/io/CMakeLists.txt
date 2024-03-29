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
set(DB_LIB_PATHS "/usr/local/lib" CACHE PATH "Paths to Hiredis/RocksDB lib")

file(GLOB local_fs_test_src
  local_fs_test.cpp
)
add_executable(local_fs_test ${local_fs_test_src})
target_compile_features(local_fs_test PUBLIC cxx_std_17)
target_link_libraries(local_fs_test PUBLIC huge_ctr_shared gtest gtest_main stdc++fs)

if (ENABLE_HDFS AND NOT DISABLE_CUDF)
  file (GLOB hdfs_backend_test_src
    hdfs_backend_test.cpp
  )
  file(GLOB file_loader_test_src
    file_loader_test.cpp
  )
  add_executable(hdfs_backend_test ${hdfs_backend_test_src})
  add_executable(file_loader_test ${file_loader_test_src})
  target_compile_features(hdfs_backend_test PUBLIC cxx_std_17)
  target_link_libraries(hdfs_backend_test PUBLIC huge_ctr_shared gtest gtest_main stdc++fs)
  target_link_libraries(file_loader_test PUBLIC huge_ctr_shared gtest gtest_main)
endif()

if (ENABLE_S3 AND NOT DISABLE_CUDF)
  file(GLOB s3_backend_test_src
    s3_backend_test.cpp
  )
  add_executable(s3_backend_test ${s3_backend_test_src})
  target_compile_features(s3_backend_test PUBLIC cxx_std_17)
  target_link_libraries(s3_backend_test PUBLIC huge_ctr_shared gtest gtest_main stdc++fs)
endif()

if (ENABLE_GCS AND NOT DISABLE_CUDF)
  file(GLOB gcs_backend_test_src
    gcs_backend_test.cpp
  )
  add_executable(gcs_backend_test ${gcs_backend_test_src})
  target_compile_features(gcs_backend_test PUBLIC cxx_std_17)
  target_link_libraries(gcs_backend_test PUBLIC huge_ctr_shared gtest gtest_main stdc++fs)
endif()

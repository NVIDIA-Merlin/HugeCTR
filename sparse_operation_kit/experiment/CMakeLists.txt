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

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/variable/impl/dynamic_embedding_table/
    ${CMAKE_CURRENT_SOURCE_DIR}/variable/impl/dynamic_embedding_table/cuCollections/include
    ${CMAKE_CURRENT_SOURCE_DIR}/variable/impl/dynamic_embedding_table/cudf
)

file(GLOB files
    ${CMAKE_CURRENT_SOURCE_DIR}/lookup/ops/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lookup/kernels/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lookup/impl/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/lookup/impl/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/variable/ops/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/variable/kernels/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/variable/impl/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/variable/impl/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/variable/impl/dynamic_embedding_table/dynamic_embedding_table.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/variable/impl/dynamic_embedding_table/hash_table.cu
)

if (ENABLE_DEEPREC)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/third_party)

execute_process (
    COMMAND bash -c "rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/external/third_party/*"
    OUTPUT_VARIABLE CLEAN_EXTERNAL_REPO
)
message(STATUS CLEAN_EXTERNAL_REPO=${CLEAN_EXTERNAL_REPO})

execute_process (
    COMMAND bash -c "cp -r $ENV{DeepRecBuild}/external/com_github_google_leveldb ${CMAKE_CURRENT_SOURCE_DIR}/external/third_party/leveldb"
    OUTPUT_VARIABLE COPY_LEVELDB
)
message(STATUS "COPY_LEVELDB=${COPY_LEVELDB}")

execute_process (
    COMMAND bash -c "cp -r $ENV{DeepRecBuild}/external/sparsehash_c11 ${CMAKE_CURRENT_SOURCE_DIR}/external/third_party/dense_hash_map"
    OUTPUT_VARIABLE COPY_DENSE_HASH_MAP
)
message(STATUS "COPY_DENSE_HASH_MAP=${COPY_DENSE_HASH_MAP}")

execute_process (
    COMMAND bash -c "cp -r $ENV{DeepRecBuild}/external/cuCollections/include/ ${CMAKE_CURRENT_SOURCE_DIR}/external/third_party/cuco_hash_table"
    OUTPUT_VARIABLE COPY_CUCO_HASH_MAP
)
message(STATUS "COPY_CUCO_HASH_MAP=${COPY_CUCO_HASH_MAP}")

include_directories(
    $ENV{DeepRecWorkdir}/addons/sparse_operation_kit/core/adapter/
    ${CMAKE_CURRENT_SOURCE_DIR}/external/
    ${CMAKE_CURRENT_SOURCE_DIR}/external/third_party/dense_hash_map
    ${CMAKE_CURRENT_SOURCE_DIR}/external/third_party/leveldb/include
)

list(APPEND files $ENV{DeepRecWorkdir}/addons/sparse_operation_kit/core/adapter/lookup_adapter.cpp)
endif()

add_library(sok_experiment SHARED ${files})
target_link_libraries(sok_experiment PUBLIC ${TF_LINK_FLAGS} core embedding hugectr_core23)

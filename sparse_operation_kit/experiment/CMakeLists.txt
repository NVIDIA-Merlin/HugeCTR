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

add_library(sok_experiment SHARED ${files})
target_link_libraries(sok_experiment PUBLIC ${TF_LINK_FLAGS} core embedding)

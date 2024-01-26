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
file(GLOB dlrm_raw_src
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/base/debug/logger.cpp
  dlrm_raw.cu
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

set(CONDA_PREFIX /opt/conda)

include_directories(${CONDA_PREFIX}/include)
include_directories(${PROJECT_SOURCE_DIR}/tools/dlrm_script)
include_directories(${PROJECT_SOURCE_DIR}/third_party/cub)
include_directories(${CONDA_PREFIX}/include/libcudf/libcudacxx)

link_directories(${CONDA_PREFIX}/lib)

# Check cudf version
execute_process(COMMAND bash -c "pip show cudf|grep Version | sed 's/.*: //'" OUTPUT_VARIABLE CUDF_VERSION)
message(STATUS "CUDF_VERSION = ${CUDF_VERSION}")

string(COMPARE EQUAL "${CUDF_VERSION}" "" CUDF_RESULT)
if (NOT CUDF_RESULT)
    string(REPLACE "." ";" CUDF_VERSION_LIST ${CUDF_VERSION})
    list(GET CUDF_VERSION_LIST 0 CUDF_VERSION_MAJOR)
    list(GET CUDF_VERSION_LIST 1 CUDF_VERSION_MINOR)
    list(GET CUDF_VERSION_LIST 1 CUDF_VERSION_PATCH)
    add_compile_definitions(CUDF_VERSION_MAJOR=${CUDF_VERSION_MAJOR})
    add_compile_definitions(CUDF_VERSION_MINOR=${CUDF_VERSION_MINOR})
    if(${CUDF_VERSION_MAJOR} EQUAL 23 AND ${CUDF_VERSION_MINOR} GREATER 6)
       add_definitions(-DCUDF_GE_2306)
    endif()
else()
    message(FATAL_ERROR "Can not detect cudf in your environment! ")
endif()

add_executable(dlrm_raw ${dlrm_raw_src})

target_link_libraries(dlrm_raw PUBLIC hugectr_core23)
if(MPI_FOUND)
  target_link_libraries(dlrm_raw PUBLIC ${CUDART_LIB} cudf ${MPI_CXX_LIBRARIES})
else()
  target_link_libraries(dlrm_raw PUBLIC ${CUDART_LIB} cudf)
endif()
target_compile_features(dlrm_raw PUBLIC cxx_std_17)




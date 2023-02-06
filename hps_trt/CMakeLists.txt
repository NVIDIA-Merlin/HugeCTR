#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hpsPlugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakes)

# setting environment variables
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

if(NOT MSVC)
  set_ifndef(TRT_LIB_DIRS /usr/lib/x86_64-linux-gnu)
  set_ifndef(TRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu) 
endif()

find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIRS} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

# applying clang format
option(CLANGFORMAT "Setup clangformat target" ON)
if(CLANGFORMAT)
  include(ClangFormat)
  file(GLOB_RECURSE HPS_PLUGIN_SRC
    ${PROJECT_SOURCE_DIR}/../HugeCTR/src/hps/*.cpp
    ${PROJECT_SOURCE_DIR}/../HugeCTR/src/hps/*.cc
    ${PROJECT_SOURCE_DIR}/../HugeCTR/src/hps/*.cu
    ${PROJECT_SOURCE_DIR}/../HugeCTR/include/hps/*.hpp
    ${PROJECT_SOURCE_DIR}/../HugeCTR/include/hps/*.h
    ${PROJECT_SOURCE_DIR}/../HugeCTR/include/hps/*.cuh
    ${PROJECT_SOURCE_DIR}/hpsPlugin/*.cpp
    ${PROJECT_SOURCE_DIR}/hpsPlugin/*.cc
    ${PROJECT_SOURCE_DIR}/hpsPlugin/*.cu
    ${PROJECT_SOURCE_DIR}/hpsPlugin/*.hpp
    ${PROJECT_SOURCE_DIR}/hpsPlugin/*.h
    ${PROJECT_SOURCE_DIR}/hpsPlugin/*.cuh
  )
  set(clangformat_srcs ${HPS_PLUGIN_SRC})
  clangformat_setup("${clangformat_srcs}")
endif()

find_package(CUDA REQUIRED)

# setting compiler flags
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Setting default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

foreach(arch_name ${SM})
    if (arch_name STREQUAL 80 OR 
        arch_name STREQUAL 75 OR 
        arch_name STREQUAL 70)
        list(APPEND cuda_arch_list ${arch_name})
        message(STATUS "Assign GPU architecture (sm=${arch_name})")
    else()
        message(FATAL_ERROR "Unknown or unsupported GPU architecture (set sm=70)")
    endif()
endforeach()

list(LENGTH cuda_arch_list cuda_arch_list_length)
if(${cuda_arch_list_length} EQUAL 0)
    list(APPEND cuda_arch_list "70")
    message(STATUS "Assign default GPU architecture sm=70")
endif()
list(REMOVE_DUPLICATES cuda_arch_list)

foreach(arch_name ${cuda_arch_list})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch_name},code=sm_${arch_name}")
endforeach()

if (${CMAKE_BUILD_TYPE} MATCHES "Release")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -Wall -Werror -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -Xcompiler -Wall,-Werror,-Wno-error=cpp,-Wno-error=parentheses -Xcudafe --display_error_number -Xcudafe --diag_suppress=177")
else()
    set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG} -O0 -g -Wall -Werror -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall -Werror -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall,-Werror,-Wno-error=cpp,-Wno-error=parentheses,-Wno-sign-compare")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# headers
include_directories(
    ${PROJECT_SOURCE_DIR}/hps_cc/hps_cc_infra/include/
    ${PROJECT_SOURCE_DIR}/hps_cc/framework/kernels/
    ${PROJECT_SOURCE_DIR}/../
    ${PROJECT_SOURCE_DIR}/../HugeCTR/
    ${PROJECT_SOURCE_DIR}/../HugeCTR/include
    ${PROJECT_SOURCE_DIR}/../third_party
    ${PROJECT_SOURCE_DIR}/../third_party/parallel-hashmap
    ${PROJECT_SOURCE_DIR}/../third_party/json/single_include
    ${PROJECT_SOURCE_DIR}/../third_party/xxhash
    ${PROJECT_SOURCE_DIR}/../gpu_cache/include
    ${PROJECT_SOURCE_DIR}/../third_party/redis_pp/src
    ${PROJECT_SOURCE_DIR}/../third_party/redis_pp/src/sw/redis++/tls
    ${PROJECT_SOURCE_DIR}/../third_party/redis_pp/src/sw/redis++/cxx17
    ${PROJECT_SOURCE_DIR}/../third_party/rocksdb/include
    ${PROJECT_SOURCE_DIR}/../third_party/librdkafka/src
    ${CUDA_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIRS}
)

add_definitions(-DTENSORRT_BUILD_LIB)

install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION hps_trt)
install(DIRECTORY ${CMAKE_BINARY_DIR}/bin DESTINATION hps_trt)

# building third party libraries
configure_file(${PROJECT_SOURCE_DIR}/../HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/../HugeCTR/include/config.hpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/../third_party third_party)

# building HPS C++ source code
add_subdirectory(${PROJECT_SOURCE_DIR}/../gpu_cache/src gpu_cache)
add_subdirectory(${PROJECT_SOURCE_DIR}/../HugeCTR/src/hps hps)

# building hps tensorrt plugin
file(GLOB_RECURSE HPS_PLUGIN_SRC
    ${CMAKE_SOURCE_DIR}/hpsPlugin/hpsPlugin.cpp
    # ${CMAKE_SOURCE_DIR}/hpsPlugin/facade.cc
    # ${CMAKE_SOURCE_DIR}/hpsPlugin/lookup_manager.cc
)
add_library(hps_plugin MODULE ${HPS_PLUGIN_SRC})

target_link_libraries(hps_plugin PRIVATE ${NVINFER_LIB})
target_link_libraries(hps_plugin PUBLIC huge_ctr_hps ${CUDA_LIBRARIES})

target_compile_features(hps_plugin PUBLIC cxx_std_17)
set_target_properties(hps_plugin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(hps_plugin PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(hps_plugin PROPERTIES CUDA_ARCHITECTURES OFF)


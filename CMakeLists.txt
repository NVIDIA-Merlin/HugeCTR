# 
# Copyright (c) 2020, NVIDIA CORPORATION.
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


cmake_minimum_required(VERSION 3.8)
project(HugeCTR LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

option(CLANGFORMAT "Setup clangformat target" OFF)
if(CLANGFORMAT)
  include(ClangFormat)

  file(GLOB_RECURSE HUGECTR_SRC
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.hpp
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.cpp
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.cu
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.cuh
    ${PROJECT_SOURCE_DIR}/utest/*.hpp
    ${PROJECT_SOURCE_DIR}/utest/*.cpp
    ${PROJECT_SOURCE_DIR}/utest/*.cu
    ${PROJECT_SOURCE_DIR}/utest/*.cuh
  )

  set(clangformat_srcs ${HUGECTR_SRC})
  clangformat_setup("${clangformat_srcs}")
endif()


option(NCCL_A2A "NCCL all2all mode: use NCCL for all2all communication" ON)
if (NCCL_A2A) 
  message(STATUS "-- NCCL_A2A is ON")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DNCCL_A2A")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DNCCL_A2A")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNCCL_A2A")
endif()

find_package(CUDA REQUIRED)
find_package(CUDNN REQUIRED)
find_package(NCCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads)

option(ENABLE_MULTINODES "Enable multi-nodes training" OFF)
if(ENABLE_MULTINODES)
message(STATUS "Multi Node Enabled")
find_package(MPI)
find_package(UCX)
find_package(HWLOC)
endif()

set(CUDA_SEPARABLE_COMPILATION ON) 

if (OPENMP_FOUND)
message(STATUS "OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(JSON_BuildTests OFF CACHE INTERNAL "")

option(VAL_MODE "Validation mode: set determined mode for data reader and csv format for loss print" OFF)
if (VAL_MODE) 
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DVAL")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DVAL")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DVAL")
endif()

# setting compiler flags
foreach(arch_name ${SM})
    if (arch_name STREQUAL 80 OR 
        arch_name STREQUAL 75 OR 
        arch_name STREQUAL 70)
        list(APPEND cuda_arch_list ${arch_name})
        message(STATUS "-- Assign GPU architecture (sm=${arch_name})")
    else()
        message(FATAL_ERROR "-- Unknown or unsupported GPU architecture (set sm=70)")
    endif()
endforeach()

list(LENGTH cuda_arch_list cuda_arch_list_length)
if(${cuda_arch_list_length} EQUAL 0)
    list(APPEND cuda_arch_list "70")
endif()
list(REMOVE_DUPLICATES cuda_arch_list)

foreach(arch_name ${cuda_arch_list})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch_name},code=sm_${arch_name}")
endforeach()

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -Wall -Werror")	
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true --Werror all-warnings -Xcompiler -Wall -Xcompiler -Werror")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G --Werror all-warnings -Xcompiler -Wall -Xcompiler -Werror")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

configure_file(${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp)

# building
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/test
  ${PROJECT_SOURCE_DIR}/HugeCTR/include
  ${CUDA_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/submodules/cutlass
  ${PROJECT_SOURCE_DIR}/third_party/cuml/cpp
  ${PROJECT_SOURCE_DIR}/third_party/cuml/cpp/include
  ${PROJECT_SOURCE_DIR}/third_party/cuml/cpp/src_prims
  ${PROJECT_SOURCE_DIR}/HugeCTR
  ${PROJECT_SOURCE_DIR}/HugeCTR/include
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cub
  ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include
  ${PROJECT_SOURCE_DIR}/third_party/json/single_include

  ${CUDNN_INC_PATHS}
  ${NCCL_INC_PATHS}
  ${HWLOC_INC_PATHS}
  ${UCX_INC_PATHS}
  $ENV{CONDA_PREFIX}/include)

if(ENABLE_MULTINODES)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DENABLE_MPI")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DENABLE_MPI")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_MPI")
  include_directories(${MPI_INCLUDE_PATH})
  include_directories(${HWLOC_INC_PATHS})
  include_directories(${UCX_INC_PATHS})
endif()

if(OPENMP_FOUND)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
  message(STATUS "add -fopenmp to compiler")
else()
  message(FATAL_ERROR "without openmp the multi-node all2all will fail")
endif()

link_directories(
  ${CUDNN_LIB_PATHS}
  ${NCCL_LIB_PATHS}
  ${HWLOC_LIB_PATHS}
  ${UCX_LIB_PATHS}
  $ENV{CONDA_PREFIX}/lib)

#setting binary files install path
install(DIRECTORY ${CMAKE_BINARY_DIR}/bin DESTINATION /usr/local/hugectr)

add_subdirectory(HugeCTR/src)
add_subdirectory(third_party/googletest)
add_subdirectory(third_party/json)
add_subdirectory(test/utest)
add_subdirectory(tools)

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
project(HugeCTR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

option(CLANGFORMAT "Setup clangformat target" ON)
if(CLANGFORMAT)
  include(ClangFormat)

  file(GLOB_RECURSE HUGECTR_SRC
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.hpp
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.cpp
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.cu
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.cuh
    ${PROJECT_SOURCE_DIR}/test/utest/*.hpp
    ${PROJECT_SOURCE_DIR}/test/utest/*.cpp
    ${PROJECT_SOURCE_DIR}/test/utest/*.cu
    ${PROJECT_SOURCE_DIR}/test/utest/*.cuh
    ${PROJECT_SOURCE_DIR}/tools/*.hpp
    ${PROJECT_SOURCE_DIR}/tools/*.cpp
    ${PROJECT_SOURCE_DIR}/tools/*.cu
    ${PROJECT_SOURCE_DIR}/tools/*.cuh
  )

  set(clangformat_srcs ${HUGECTR_SRC})
  clangformat_setup("${clangformat_srcs}")
endif()

option(DISABLE_A2A_WARMUP "Disable nccl a2a warm up" OFF)
if (DISABLE_A2A_WARMUP)
  message(STATUS "-- DISABLE_A2A_WARMUP is ON")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DDISABLE_A2A_WARMUP")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DDISABLE_A2A_WARMUP")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDISABLE_A2A_WARMUP")
endif()

option(DISABLE_CUDF "Disable cudf: disable parquet format related features" OFF)
if (DISABLE_CUDF)
  message(STATUS "-- DISABLE_CUDF is ON")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DDISABLE_CUDF")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DDISABLE_CUDF")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDISABLE_CUDF")
endif()

option(SHARP_A2A "Enable SHARP All2All" OFF)
if (SHARP_A2A)
  message (STATUS "-- SHARP_A2A is ON")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DSHARP_A2A")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DSHARP_A2A")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSHARP_A2A")
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
  find_package(SHARP)
endif()

set(CUDA_SEPARABLE_COMPILATION ON)

if (OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(JSON_BuildTests OFF CACHE INTERNAL "")

option(KEY_HIT_RATIO "Verbose mode of embedding training cache: print out the key hit ratio" OFF)
if (KEY_HIT_RATIO)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DKEY_HIT_RATIO")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DKEY_HIT_RATIO")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DKEY_HIT_RATIO")
endif()


# setting compiler flags
foreach(arch_name ${SM})
    if (arch_name STREQUAL 80 OR
        arch_name STREQUAL 75 OR
        arch_name STREQUAL 70 OR
        arch_name STREQUAL 61 OR
        arch_name STREQUAL 60)
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
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unknown-pragmas")
#TODO: the warning should not be suppresed in the  long term
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -Xcompiler -Wall,-Werror,-Wno-error=cpp,-Wno-error=parentheses -Xcudafe --display_error_number -Xcudafe --diag_suppress=177")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG} -O0 -g -Wall -Werror")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall -Werror -Wno-unknown-pragmas -Wno-sign-compare")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall,-Werror,-Wno-error=cpp,-Wno-error=parentheses,-Wno-sign-compare")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(ENABLE_MULTINODES)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DENABLE_MPI")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DENABLE_MPI")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_MPI")
  include_directories(${MPI_INCLUDE_PATH})
  include_directories(${HWLOC_INC_PATHS})
  include_directories(${UCX_INC_PATHS})
  link_directories(/opt/hpcx/ompi/lib)
endif()

option(ENABLE_PROFILING "Profiling mode: whether to turn on inline profiling" OFF)
if(ENABLE_PROFILING)
  message("STATUS" "-- PROFILING is ON")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DENABLE_PROFILING")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DENABLE_PROFILING")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_PROFILING")
  link_libraries(cuda)
endif()

configure_file(${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp)

# building
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/test
  ${PROJECT_SOURCE_DIR}/HugeCTR/include
  $ENV{CONDA_PREFIX}/include
  $ENV{CONDA_PREFIX}/include/libcudf/libcudacxx
  ${CUDA_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/third_party/cuml/cpp
  ${PROJECT_SOURCE_DIR}/third_party/cuml/cpp/include
  ${PROJECT_SOURCE_DIR}/third_party/cuml/cpp/src_prims
  ${PROJECT_SOURCE_DIR}/HugeCTR
  ${PROJECT_SOURCE_DIR}/HugeCTR/include
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cutlass
  ${PROJECT_SOURCE_DIR}/third_party/cpptqdm
  ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include
  ${PROJECT_SOURCE_DIR}/third_party/googletest/googlemock/include
  ${PROJECT_SOURCE_DIR}/third_party/json/single_include
  ${PROJECT_SOURCE_DIR}/third_party/parallel-hashmap
  ${PROJECT_SOURCE_DIR}/third_party/xxhash
  ${PROJECT_SOURCE_DIR}/gpu_cache/include

  ${CUDNN_INC_PATHS}
  ${NCCL_INC_PATHS}
  ${HWLOC_INC_PATHS}
  ${UCX_INC_PATHS}
  ${SHARP_INC_PATHS})

if (CUDA_VERSION_MAJOR LESS 11)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/cub)
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
  ${SHARP_LIB_PATHS}
  $ENV{CONDA_PREFIX}/lib)

#setting python interface file install path
install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION hugectr)
#install inference header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/HugeCTR/include/inference DESTINATION hugectr/include)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/HugeCTR/include/hps DESTINATION hugectr/include)
add_subdirectory(third_party/json)
install(DIRECTORY ${CMAKE_BINARY_DIR}/bin DESTINATION hugectr)

add_subdirectory(gpu_cache/src)
add_subdirectory(gpu_cache/test)

option(ENABLE_HDFS "Enable HDFS" OFF)
if(ENABLE_HDFS)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}    -DENABLE_HDFS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}    -DENABLE_HDFS")
endif()

option(ENABLE_INFERENCE "Enable Inference" OFF)
if(ENABLE_INFERENCE)
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DENABLE_INFERENCE")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DENABLE_INFERENCE")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_INFERENCE")
add_subdirectory(third_party/googletest)
add_subdirectory(HugeCTR/src/hps)
add_subdirectory(HugeCTR/src/inference)
add_subdirectory(HugeCTR/src/cpu)
add_subdirectory(test/utest/hps)
add_subdirectory(test/utest/inference)
else()
#setting binary files install path
add_subdirectory(HugeCTR/src)
option(ENABLE_HCTR "Build core HCTR backend" ON)
add_subdirectory(HugeCTR/core)
add_subdirectory(HugeCTR/embedding)
add_subdirectory(third_party/googletest)
add_subdirectory(test/utest)
add_subdirectory(tools)
endif()

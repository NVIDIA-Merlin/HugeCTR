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

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(hpsTorch LANGUAGES CXX CUDA)

set(Torch_DIR /usr/local/lib/${PYTHON_VERSION}/dist-packages/torch/share/cmake/Torch)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakes)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}")

# Step 1) Clangformat source code.
option(CLANGFORMAT "Setup clangformat target" ON)
if(CLANGFORMAT)
  include(ClangFormat)
  file(GLOB_RECURSE HPS_PLUGIN_SRC
    ${PROJECT_SOURCE_DIR}/../gpu_cache/include/*.h
    ${PROJECT_SOURCE_DIR}/../gpu_cache/include/*.hpp
    ${PROJECT_SOURCE_DIR}/../gpu_cache/include/*.cuh
    ${PROJECT_SOURCE_DIR}/../gpu_cache/src/*.cu
    ${PROJECT_SOURCE_DIR}/../HugeCTR/include/*.hpp
    ${PROJECT_SOURCE_DIR}/../HugeCTR/include/*.h
    ${PROJECT_SOURCE_DIR}/../HugeCTR/include/*.cuh
    ${PROJECT_SOURCE_DIR}/../HugeCTR/src/hps/*.cpp
    ${PROJECT_SOURCE_DIR}/../HugeCTR/src/hps/*.cu
    ${PROJECT_SOURCE_DIR}/../HugeCTR/include/hps/*.hpp
    ${PROJECT_SOURCE_DIR}/../HugeCTR/include/hps/*.cuh
    ${PROJECT_SOURCE_DIR}/*.cpp
  )
  set(clangformat_srcs ${HPS_PLUGIN_SRC})
  clangformat_setup("${clangformat_srcs}")
endif()

# Step 2) Build dependencies before we adjust the compiler configuration.
add_subdirectory(${PROJECT_SOURCE_DIR}/../third_party third_party)

# Step 3) Configure and build HugeCTR HPS + TORCH plugin.
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DSPDLOG_HEADER_ONLY -DSPDLOG_FMT_EXTERNAL -DFMT_HEADER_ONLY")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DSPDLOG_HEADER_ONLY -DSPDLOG_FMT_EXTERNAL -DFMT_HEADER_ONLY")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSPDLOG_HEADER_ONLY -DSPDLOG_FMT_EXTERNAL -DFMT_HEADER_ONLY")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# setting compiler flags
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Setting default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

if (OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

foreach(arch_name ${SM})
    if (arch_name STREQUAL 90 OR
        arch_name STREQUAL 80 OR
        arch_name STREQUAL 75 OR
        arch_name STREQUAL 70)
        message(STATUS "${arch_name} is added to generate device code")
        list(APPEND cuda_arch_list ${arch_name}-real)
    elseif (arch_name STREQUAL 61 OR
            arch_name STREQUAL 60)
        message(WARNING "The specified architecture ${arch_name} is excluded because it is not supported")
    else()
        message(FATAL_ERROR "${arch_name} is an invalid or unsupported architecture")
    endif()
endforeach()

list(REMOVE_DUPLICATES cuda_arch_list)
list(LENGTH cuda_arch_list cuda_arch_list_length)
if(${cuda_arch_list_length} EQUAL 0)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    list(APPEND cuda_arch_list 80-real)
    set(CMAKE_CUDA_ARCHITECTURES ${cuda_arch_list})
  endif()
else()
  set(CMAKE_CUDA_ARCHITECTURES ${cuda_arch_list})
endif()

message(STATUS "Target GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (${CMAKE_BUILD_TYPE} MATCHES "Release")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -Wall -Werror -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Werror,-Wno-error=cpp,-Wno-error=parentheses -Xcudafe --display_error_number -Xcudafe --diag_suppress=177")
else()
    set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG} -O0 -g -Wall -Werror -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall -Werror -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall,-Werror,-Wno-error=cpp,-Wno-error=parentheses,-Wno-sign-compare")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

if(OPENMP_FOUND)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
endif()

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# setting install folder
if(SKBUILD)
    # if we're building with skbuild, then we shouldn't be installing to /usr/local/lib
    # install the libraries alongside the python source in hps_torch/lib instead
    install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION hps_torch)
else()
    install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION /usr/local)
endif()

# headers
include_directories(
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
    ${PROJECT_SOURCE_DIR}/../third_party/embedding_cache/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# building third party libraries
configure_file(${PROJECT_SOURCE_DIR}/../HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/../HugeCTR/include/config.hpp)

# building HPS C++ source code
add_subdirectory(${PROJECT_SOURCE_DIR}/../gpu_cache/src gpu_cache)
add_subdirectory(${PROJECT_SOURCE_DIR}/../HugeCTR/src/hps hps)

# building hps torch plugin
file(GLOB_RECURSE HPS_PLUGIN_SRC
    ${CMAKE_SOURCE_DIR}/hps_torch_plugin.cpp
)
add_library(hps_torch MODULE ${HPS_PLUGIN_SRC})

target_link_libraries(hps_torch PUBLIC huge_ctr_hps ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so ${TORCH_LIBRARIES} CUDA::cudart_static ${Python_LIBRARIES})
target_compile_features(hps_torch PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(hps_torch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


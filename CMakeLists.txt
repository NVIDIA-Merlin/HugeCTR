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
project(HugeCTR LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

option(CLANGFORMAT "Setup clangformat target" ON)
if(CLANGFORMAT)
  include(ClangFormat)

  file(GLOB_RECURSE HUGECTR_SRC
    ${PROJECT_SOURCE_DIR}/benchmarks/*.hpp
    ${PROJECT_SOURCE_DIR}/benchmarks/*.cpp
    ${PROJECT_SOURCE_DIR}/benchmarks/*.cu
    ${PROJECT_SOURCE_DIR}/benchmarks/*.cuh
    ${PROJECT_SOURCE_DIR}/gpu_cache/include.hpp
    ${PROJECT_SOURCE_DIR}/gpu_cache/include/*.cuh
    ${PROJECT_SOURCE_DIR}/gpu_cache/src/*.cpp
    ${PROJECT_SOURCE_DIR}/gpu_cache/src/*.cu
    ${PROJECT_SOURCE_DIR}/hps_tf/*.hpp
    ${PROJECT_SOURCE_DIR}/hps_tf/*.cpp
    ${PROJECT_SOURCE_DIR}/hps_tf/*.cu
    ${PROJECT_SOURCE_DIR}/hps_tf/*.cuh
    ${PROJECT_SOURCE_DIR}/hps_trt/*.hpp
    ${PROJECT_SOURCE_DIR}/hps_trt/*.cpp
    ${PROJECT_SOURCE_DIR}/hps_trt/*.cu
    ${PROJECT_SOURCE_DIR}/hps_trt/*.cuh
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.hpp
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.cpp
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.cu
    ${PROJECT_SOURCE_DIR}/HugeCTR/*.cuh
    ${PROJECT_SOURCE_DIR}/test/*.hpp
    ${PROJECT_SOURCE_DIR}/test/*.cpp
    ${PROJECT_SOURCE_DIR}/test/*.cu
    ${PROJECT_SOURCE_DIR}/test/*.cuh
    ${PROJECT_SOURCE_DIR}/tools/*.hpp
    ${PROJECT_SOURCE_DIR}/tools/*.cpp
    ${PROJECT_SOURCE_DIR}/tools/*.cu
    ${PROJECT_SOURCE_DIR}/tools/*.cuh
    ${PROJECT_SOURCE_DIR}/third_party/dynamic_embedding_table/*.hpp
    ${PROJECT_SOURCE_DIR}/third_party/dynamic_embedding_table/*.cpp
    ${PROJECT_SOURCE_DIR}/third_party/dynamic_embedding_table/*.cu
    ${PROJECT_SOURCE_DIR}/third_party/dynamic_embedding_table/*.cuh
    ${PROJECT_SOURCE_DIR}/third_party/dynamic_embedding_table/*.inl
  )

  set(clangformat_srcs ${HUGECTR_SRC})
  clangformat_setup("${clangformat_srcs}")
endif()

option(ENABLE_GCS "Enable GCS" OFF)

# List of supported features.
set(HCTR_SUPPORTED_FEATURES "")
list(APPEND HCTR_SUPPORTED_FEATURES "redis")
list(APPEND HCTR_SUPPORTED_FEATURES "rocks_db")

# List of features enabled in this build configuration.
set(HCTR_FEATURES "${HCTR_SUPPORTED_FEATURES}" CACHE STRING "Enabled optional features")
list(REMOVE_DUPLICATES HCTR_FEATURES)
string(TOLOWER "${HCTR_FEATURES}" HCTR_FEATURES)

# Verify and print feature selection.
message(STATUS "Enabled optional HugeCTR features:")
foreach(HCTR_FEATURE ${HCTR_FEATURES})
  message(STATUS "  ${HCTR_FEATURE}")
  list(FIND HCTR_SUPPORTED_FEATURES "${HCTR_FEATURE}" HCTR_FEATURE_ENABLED)
  if(${HCTR_FEATURE_ENABLED} EQUAL -1)
    message(FATAL_ERROR "Unsupported feature \"${HCTR_FEATURE}\"! Supported features: ${HCTR_SUPPORTED_FEATURES}")
  endif()
endforeach()

# Build dependencies.
add_subdirectory(third_party)

# Expand include path and library list accordingly, and toggle preprocessor flags.
foreach(HCTR_FEATURE ${HCTR_FEATURES})
  if(${HCTR_FEATURE} STREQUAL "redis")
    include_directories(
      ${PROJECT_SOURCE_DIR}/third_party/redis_pp/src
      ${PROJECT_SOURCE_DIR}/third_party/redis_pp/src/sw/redis++/cxx17
      ${PROJECT_SOURCE_DIR}/third_party/redis_pp/src/sw/redis++/tls
    )
    link_libraries(hiredis redis++)
  elseif(${HCTR_FEATURE} STREQUAL "rocks_db")
    include_directories(
      ${PROJECT_SOURCE_DIR}/third_party/rocksdb/include
    )
    link_libraries(rocksdb-shared)
  endif()

  string(TOUPPER "${HCTR_FEATURE}" HCTR_FEATURE)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DHCTR_USE_${HCTR_FEATURE}")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DHCTR_USE_${HCTR_FEATURE}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DHCTR_USE_${HCTR_FEATURE}")
endforeach()

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DSPDLOG_HEADER_ONLY -DSPDLOG_FMT_EXTERNAL -DFMT_HEADER_ONLY")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DSPDLOG_HEADER_ONLY -DSPDLOG_FMT_EXTERNAL -DFMT_HEADER_ONLY")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSPDLOG_HEADER_ONLY -DSPDLOG_FMT_EXTERNAL -DFMT_HEADER_ONLY")

option(DISABLE_A2A_WARMUP "Disable nccl a2a warm up" OFF)
if (DISABLE_A2A_WARMUP)
  message(STATUS "-- DISABLE_A2A_WARMUP is ON")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DDISABLE_A2A_WARMUP")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DDISABLE_A2A_WARMUP")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDISABLE_A2A_WARMUP")
endif()

option(DISABLE_CUDF "Disable cudf: disable parquet format related features" OFF)
# this manual definition is a WAR and RMM team will fix it in the future
add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
option(USE_CUDART_STATIC "Setup clangformat target" OFF)
if(USE_CUDART_STATIC)
  set(DISABLE_CUDF ON)
  set(CUDART_LIB CUDA::cudart_static)
else()
  set(CUDART_LIB CUDA::cudart)
endif()

if (DISABLE_CUDF)
  message(STATUS "-- DISABLE_CUDF is ON")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DDISABLE_CUDF")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DDISABLE_CUDF")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDISABLE_CUDF")
else()
  execute_process(
    COMMAND bash -c "pip show cudf|grep Version | sed 's/.*: //'" 
    OUTPUT_VARIABLE CUDF_VERSION
  )

  string(REPLACE "." ";" VERSION_LIST ${CUDF_VERSION})
  list(GET VERSION_LIST 0 CUDF_VERSION_MAJOR)
  list(GET VERSION_LIST 1 CUDF_VERSION_MINOR)
  # list(GET VERSION_LIST 2 CUDF_VERSION_PATCH)
  # add_compile_definitions(CUDF_VERSION_PATCH=${CUDF_VERSION_PATCH})

  add_compile_definitions(CUDF_VERSION_MAJOR=${CUDF_VERSION_MAJOR})
  add_compile_definitions(CUDF_VERSION_MINOR=${CUDF_VERSION_MINOR})
  message(STATUS "CUDF_VERSION is  ${CUDF_VERSION}")

  find_package(Parquet REQUIRED CONFIG PATHS  /usr/lib/cmake/arrow/ /usr/lib/cmake/Parquet/ NO_DEFAULT_PATH)
  if(Parquet_FOUND AND NOT ENABLE_HDFS AND NOT ENABLE_S3 AND NOT ENABLE_GCS)
    message (STATUS "Arrow Parquet is found")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DENABLE_ARROW_PARQUET")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DENABLE_ARROW_PARQUET")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_ARROW_PARQUET")
  endif()

endif()

option(SHARP_A2A "Enable SHARP All2All" OFF)
if (SHARP_A2A)
  message (STATUS "-- SHARP_A2A is ON")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DSHARP_A2A")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DSHARP_A2A")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSHARP_A2A")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(CUDNN REQUIRED)
find_package(NCCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads)

option(ENABLE_MULTINODES "Enable multi-nodes training" OFF)
option(ENABLE_INFERENCE "Enable Inference" OFF)

if(ENABLE_MULTINODES AND ENABLE_INFERENCE)
  message(WARNING "Inference can be only enabled with the multi-node mode off. Set ENABLE_MULTINODES=OFF")
  set(ENABLE_MULTINODES OFF)
endif()

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

option(KEY_HIT_RATIO "Verbose mode of embedding training cache: print out the key hit ratio" OFF)
if (KEY_HIT_RATIO)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DKEY_HIT_RATIO")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DKEY_HIT_RATIO")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DKEY_HIT_RATIO")
endif()

if(ENABLE_INFERENCE)
  add_definitions(-DLIBCUDACXX_VERSION)
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

option(ENABLE_PROFILER "Enable HPS profiler" OFF)
if(ENABLE_PROFILER)
add_definitions(-DBENCHMARK_HPS)
endif()

if (USE_HUGE_PAGES)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DUSE_HUGE_PAGES")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DUSE_HUGE_PAGES")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_HUGE_PAGES")
  message(STATUS "Hugepages: Enabled")
else()
  message(STATUS "Hugepages: Disabled")
endif()

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -Wall -Werror -Wno-sign-compare")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-sign-compare")
#TODO: the warning should not be suppressed in the  long term
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Werror -Xcudafe --display_error_number -Xcudafe --diag_suppress=177")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG} -O0 -g ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G ${CMAKE_CUDA_FLAGS}")

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

configure_file(${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp)

# building
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/test
  ${PROJECT_SOURCE_DIR}/HugeCTR/include
  $ENV{CONDA_PREFIX}/include
  $ENV{CONDA_PREFIX}/include/libcudf/libcudacxx
  ${CUDAToolkit_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/third_party/cuml/cpp
  ${PROJECT_SOURCE_DIR}/third_party/cuml/cpp/include
  ${PROJECT_SOURCE_DIR}/third_party/cuml/cpp/src_prims
  ${PROJECT_SOURCE_DIR}/HugeCTR
  ${PROJECT_SOURCE_DIR}/HugeCTR/include
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/argparse/include
  ${PROJECT_SOURCE_DIR}/third_party/cutlass
  ${PROJECT_SOURCE_DIR}/third_party/cpptqdm
  ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include
  ${PROJECT_SOURCE_DIR}/third_party/googletest/googlemock/include
  ${PROJECT_SOURCE_DIR}/third_party/json/single_include
  ${PROJECT_SOURCE_DIR}/third_party/parallel-hashmap
  ${PROJECT_SOURCE_DIR}/third_party/librdkafka/src
  ${PROJECT_SOURCE_DIR}/gpu_cache/include
  ${PROJECT_SOURCE_DIR}/third_party/dynamic_embedding_table
  ${PROJECT_SOURCE_DIR}/third_party/dynamic_embedding_table/cuCollections/include
  ${PROJECT_SOURCE_DIR}/third_party/HierarchicalKV/include
  ${PROJECT_SOURCE_DIR}/third_party/embedding_cache/include
  
  ${CMAKE_BINARY_DIR}/_deps/google-cloud-cpp-src

  ${CUDNN_INC_PATHS}
  ${NCCL_INC_PATHS}
  ${HWLOC_INC_PATHS}
  ${UCX_INC_PATHS}
  ${SHARP_INC_PATHS})

if (CUDAToolkit_VERSION_MAJOR LESS 11)
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
  ${PARQUET_LIB_PATHS}
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
install(DIRECTORY ${PROJECT_SOURCE_DIR}/HugeCTR/include/inference_benchmark DESTINATION hugectr/include)
install(DIRECTORY ${CMAKE_BINARY_DIR}/bin DESTINATION hugectr)

add_subdirectory(gpu_cache/src)

option(ENABLE_HDFS "Enable HDFS" OFF)
if(ENABLE_HDFS)
  if(ENABLE_HDFS STREQUAL "MINIMAL")
    message("HDFS build mode: Client only")
  else()
    message("HDFS build mode: Full")
  endif()
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DENABLE_HDFS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_HDFS")

  # Build and Install Hadoop
  include(SetupHadoop)
  hadoop_setup(${ENABLE_HDFS})
  set(ENABLE_HDFS ON)
endif()

option(ENABLE_S3 "Enable S3" OFF)
if(ENABLE_S3)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DENABLE_S3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_S3")

  include(SetupS3)
  s3_setup()
endif()

if(ENABLE_GCS)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DENABLE_GCS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_GCS")
endif()

option(ENABLE_INFERENCE "Enable Inference" OFF)
if(ENABLE_INFERENCE)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DENABLE_INFERENCE")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DENABLE_INFERENCE")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_INFERENCE")
  add_subdirectory(HugeCTR/core23)
  add_subdirectory(HugeCTR/src/hps)
  add_subdirectory(HugeCTR/src/inference_benchmark)
  add_subdirectory(test/utest/hps)
else()
  #setting binary files install path
  add_subdirectory(HugeCTR/src)
  add_subdirectory(HugeCTR/core23)
  add_subdirectory(HugeCTR/embedding)
  add_subdirectory(test/utest)
  add_subdirectory(tools)
  add_subdirectory(benchmarks)
  add_subdirectory(test/embedding_cache_perf_test)
endif()

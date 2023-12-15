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

cmake_minimum_required(VERSION 3.20)
project(SparseOperationKit LANGUAGES CXX CUDA)

message(STATUS "Building Sparse Operation Kit from source.")

option(ENV_PYTHONPATH "keep PYTHONPATH same with os env when SKBUILD" ON)
if (ENV_PYTHONPATH)
set(ENV{PYTHONPATH} ${ENV_PYTHONPATH})
endif()
set(TF_VERSION,"")
#check TF version
execute_process(COMMAND python -c "import tensorflow as tf;print(tf.__version__)" OUTPUT_VARIABLE TF_VERSION)
message(STATUS "TF_VERSION = ${TF_VERSION}")
string(COMPARE EQUAL "${TF_VERSION}" "" TF_RESULT)
if (NOT TF_RESULT)
    string(REPLACE "." ";" TF_VERSION_LIST ${TF_VERSION})
    list(GET TF_VERSION_LIST 0 TF_VERSION_MAJOR)
    list(GET TF_VERSION_LIST 1 TF_VERSION_MINOR)
    list(GET TF_VERSION_LIST 2 TF_VERSION_PATCH)
    if(${TF_VERSION_MAJOR} GREATER 1 AND ${TF_VERSION_MINOR} GREATER 9)
        add_definitions(-DTF_GE_210)
        set_property(GLOBAL PROPERTY SOK_CXX_STANDARD_PROPERTY cxx_std_17)
        set_property(GLOBAL PROPERTY SOK_CUDA_STANDARD_PROPERTY cuda_std_17)
    else()
        add_definitions(-DTF_LESS_210)
        set_property(GLOBAL PROPERTY SOK_CXX_STANDARD_PROPERTY cxx_std_14)
        set_property(GLOBAL PROPERTY SOK_CUDA_STANDARD_PROPERTY cuda_std_14)
    endif()
 
    
    if(${TF_VERSION_MAJOR} GREATER 1 AND ${TF_VERSION_MINOR} GREATER 10)
        add_definitions(-DTF_GE_211)
    endif()


    if(${TF_VERSION_MAJOR} GREATER 1 AND ${TF_VERSION_MINOR} GREATER 11)
        add_definitions(-DTF_GE_212)
    endif()
else()
    message(FATAL_ERROR "Can not detect tensorflow in your environment,please install tensorflow(tf1 support version 1.15, for tf2 support version 2.60~latest) ")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakes)

find_package(CUDAToolkit REQUIRED)
find_package(NCCL REQUIRED)
find_package(TensorFlow REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads)

set(CUDA_SEPARABLE_COMPILATION ON)

# whether use nvtx
option(USE_NVTX "Use nvtx for profiling" OFF)
if (USE_NVTX)
    message(STATUS "Add nvtx for profiling")
    add_definitions(-DUSE_NVTX)
    find_package(NVTX REQUIRED)
endif()

option(ENABLE_HCTR "Build core HCTR backend" ON)



# setting compiler flags
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Setting default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
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

if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas")
    if (${TF_VERSION_MAJOR} EQUAL 1 AND NOT ENABLE_DEEPREC)
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-error=class-memaccess")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wno-error=cpp")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
else()
    # -------- set flags for DEBUG mode --- #
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wno-unknown-pragmas")
    if (${TF_VERSION_MAJOR} EQUAL 1)
        set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wno-error=class-memaccess")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wno-error=class-memaccess")
    endif()
    add_definitions(-DNDEBUG)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler -Wall,-Wno-error=cpp")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcudafe --display_error_number")
    # set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
    # ------------------------------------- #
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TF_COMPILE_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGOOGLE_CUDA=1 ${TF_COMPILE_FLAGS}")

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# whether build for DeepRec
# if build for DeepRec, need copy some third party library
option(ENABLE_DEEPREC "Enable op support for deeprec" OFF)
if (ENABLE_DEEPREC)
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DGOOGLE_CUDA -DTENSORFLOW_USE_GPU_EV")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DGOOGLE_CUDA -DTENSORFLOW_USE_GPU_EV")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGOOGLE_CUDA -DTENSORFLOW_USE_GPU_EV")

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/external/third_party)

execute_process (
    COMMAND bash -c "rm -rf ${PROJECT_SOURCE_DIR}/external/third_party/*"
    OUTPUT_VARIABLE CLEAN_EXTERNAL_REPO
)
message(STATUS CLEAN_EXTERNAL_REPO=${CLEAN_EXTERNAL_REPO})

execute_process (
    COMMAND bash -c "cp -r $ENV{DeepRecBuild}/external/com_github_google_leveldb ${PROJECT_SOURCE_DIR}/external/third_party/leveldb"
    OUTPUT_VARIABLE COPY_LEVELDB
)
message(STATUS "COPY_LEVELDB=${COPY_LEVELDB}")

execute_process (
    COMMAND bash -c "cp -r $ENV{DeepRecBuild}/external/sparsehash_c11 ${PROJECT_SOURCE_DIR}/external/third_party/dense_hash_map"
    OUTPUT_VARIABLE COPY_DENSE_HASH_MAP
)
message(STATUS "COPY_DENSE_HASH_MAP=${COPY_DENSE_HASH_MAP}")

execute_process (
    COMMAND bash -c "cp -r $ENV{DeepRecBuild}/external/cuCollections/include/ ${PROJECT_SOURCE_DIR}/external/third_party/cuco_hash_table"
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



# setting install folder
if(SKBUILD)
    # if we're building with skbuild, then we shouldn't be installing to /usr/local/lib
    # install the libraries alongside the python source in sparse_operation_kit/lib instead
    install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION sparse_operation_kit)
else()
    install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION /usr/local)
endif()

# headers
if(SKBUILD)
# We haven't found a proper way to maintain the directory structure of
# the parent folder(i.e. HugeCTR) when using skbuild to make pip package,
# so we use a workaround here: copy the content of parent folder into
# sparse_operation_kit/ before making pip package.
include_directories(
    ${PROJECT_SOURCE_DIR}/
    ${PROJECT_SOURCE_DIR}/../
    ${PROJECT_SOURCE_DIR}/HugeCTR/
    ${PROJECT_SOURCE_DIR}/HugeCTR/include
    ${PROJECT_SOURCE_DIR}/HugeCTR/core
    ${PROJECT_SOURCE_DIR}/HugeCTR/embedding
    ${PROJECT_SOURCE_DIR}/HugeCTR/core23
    ${PROJECT_SOURCE_DIR}/third_party/json/include
    ${PROJECT_SOURCE_DIR}/third_party/HierarchicalKV/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NCCL_INC_PATHS}
    ${MPI_INCLUDE_PATH}
)
else()
include_directories(
    ${PROJECT_SOURCE_DIR}/../
    ${PROJECT_SOURCE_DIR}/../HugeCTR/
    ${PROJECT_SOURCE_DIR}/../HugeCTR/include
    ${PROJECT_SOURCE_DIR}/../HugeCTR/core
    ${PROJECT_SOURCE_DIR}/../HugeCTR/embedding
    ${PROJECT_SOURCE_DIR}/../HugeCTR/core23
    ${PROJECT_SOURCE_DIR}/../third_party/json/include
    ${PROJECT_SOURCE_DIR}/../third_party/HierarchicalKV/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NCCL_INC_PATHS}
    ${MPI_INCLUDE_PATH}
)
endif()


# libs
link_directories(
    ${NCCL_LIB_PATHS}
    ${TF_LINK_DIR}/
    /usr/local/cuda/lib64/
)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/kit_src
    ${PROJECT_SOURCE_DIR}/kit_src/variable/impl/dynamic_embedding_table/
    ${PROJECT_SOURCE_DIR}/kit_src/variable/impl/dynamic_embedding_table/cuCollections/include
    ${PROJECT_SOURCE_DIR}/kit_src/variable/impl/dynamic_embedding_table/cudf
)

file(GLOB files
    ${PROJECT_SOURCE_DIR}/kit_src/lookup/ops/*.cc
    ${PROJECT_SOURCE_DIR}/kit_src/lookup/kernels/*.cc
    ${PROJECT_SOURCE_DIR}/kit_src/lookup/impl/*.cu
    ${PROJECT_SOURCE_DIR}/kit_src/lookup/impl/*.cc
    ${PROJECT_SOURCE_DIR}/kit_src/variable/ops/*.cc
    ${PROJECT_SOURCE_DIR}/kit_src/variable/kernels/*.cc
    ${PROJECT_SOURCE_DIR}/kit_src/variable/impl/*.cu
    ${PROJECT_SOURCE_DIR}/kit_src/variable/impl/*.cc
    ${PROJECT_SOURCE_DIR}/kit_src/variable/impl/dynamic_embedding_table/dynamic_embedding_table.cu
    ${PROJECT_SOURCE_DIR}/kit_src/variable/impl/dynamic_embedding_table/hash_table.cu
)

if(SKBUILD)
# We haven't found a proper way to maintain the directory structure of
# the parent folder(i.e. HugeCTR) when using skbuild to make pip package,
# so we use a workaround here: copy the content of parent folder into
# sparse_operation_kit/ before making pip package.
configure_file(${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp)
add_subdirectory(HugeCTR/embedding embedding.out)
add_subdirectory(HugeCTR/core23 hugectr_core23.out)
else()
configure_file(${PROJECT_SOURCE_DIR}/../HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/../HugeCTR/include/config.hpp)
add_subdirectory(../HugeCTR/embedding embedding.out)
add_subdirectory(../HugeCTR/core23 hugectr_core23.out)
endif()

# build dynamic lib
add_library(sparse_operation_kit SHARED ${files})
target_link_libraries(sparse_operation_kit PUBLIC ${TF_LINK_FLAGS} nccl CUDA::cusparse ${MPI_CXX_LIBRARIES} CUDA::curand embedding hugectr_core23)

get_property(SOK_CXX_STANDARD_FLAG GLOBAL PROPERTY SOK_CXX_STANDARD_PROPERTY)
get_property(SOK_CUDA_STANDARD_FLAG GLOBAL PROPERTY SOK_CUDA_STANDARD_PROPERTY)
target_compile_features(sparse_operation_kit PRIVATE ${SOK_CXX_STANDARD_FLAG} ${SOK_CXX_STANDARD_FLAG})
#target_compile_features(sparse_operation_kit PUBLIC cxx_std_14 cuda_std_14)

if (USE_NVTX)
    target_link_libraries(sparse_operation_kit PUBLIC ${NVTX_LIB})
endif()

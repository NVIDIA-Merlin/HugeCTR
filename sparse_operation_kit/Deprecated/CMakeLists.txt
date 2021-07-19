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
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
project(Embedding_Plugin LANGUAGES CXX CUDA)

message(STATUS "Please be noted that only embedding plugin will be built.")

set(CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -Xcompiler -Wall,-Werror,-Wno-error=cpp")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG} -O0 -Wall -Werror")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -Werror -Wno-unknown-pragmas")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall,-Werror,-Wno-error=cpp")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


set(HugeCTR_Path ${PROJECT_SOURCE_DIR}/../../)

configure_file(${HugeCTR_Path}/HugeCTR/include/config.hpp.in ${HugeCTR_Path}/HugeCTR/include/config.hpp)

# building
include_directories(
  ${HugeCTR_Path}
  ${HugeCTR_Path}/HugeCTR/include
  ${CUDA_INCLUDE_DIRS}
  ${HugeCTR_Path}/third_party/cutlass
  ${HugeCTR_Path}/third_party/cuml/cpp
  ${HugeCTR_Path}/third_party/cuml/cpp/include
  ${HugeCTR_Path}/third_party/cuml/cpp/src_prims
  ${HugeCTR_Path}/HugeCTR
  ${HugeCTR_Path}/third_party
  ${HugeCTR_Path}/third_party/cub

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

#setting python interface file install path
install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION /usr/local/hugectr)


SET(PYTHON "python3")

EXECUTE_PROCESS(
    COMMAND
        ${PYTHON} -c
        "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TF_COMPILE_FLAGS)

EXECUTE_PROCESS(
    COMMAND
        ${PYTHON} -c
        "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TF_LINK_FLAGS)

MESSAGE("-- TF LINK FLAGS = ${TF_LINK_FLAGS}")
STRING(REGEX MATCH "(^-L.*\ )" TF_LINK_DIR ${TF_LINK_FLAGS})
STRING(REPLACE "-L" "" TF_LINK_DIR ${TF_LINK_DIR})
STRING(REPLACE " " "" TF_LINK_DIR ${TF_LINK_DIR})
MESSAGE("-- TF link dir = ${TF_LINK_DIR}")

LINK_DIRECTORIES(${TF_LINK_DIR}/)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${TF_COMPILE_FLAGS}")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGOOGLE_CUDA=1 --expt-relaxed-constexpr ${TF_COMPILE_FLAGS}")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

ADD_DEFINITIONS(-DEIGEN_USE_GPU)

FILE(GLOB  hugectr_src
  ${HugeCTR_Path}/HugeCTR/src/cpu_resource.cpp
  ${HugeCTR_Path}/HugeCTR/src/gpu_resource.cpp
  ${HugeCTR_Path}/HugeCTR/src/resource_manager.cpp
  ${HugeCTR_Path}/HugeCTR/src/resource_managers/resource_manager_core.cpp
  ${HugeCTR_Path}/HugeCTR/src/data_simulator.cu
  ${HugeCTR_Path}/HugeCTR/src/optimizers/*.cu
  ${HugeCTR_Path}/HugeCTR/src/optimizer.cpp
  ${HugeCTR_Path}/HugeCTR/src/regularizer.cu
  ${HugeCTR_Path}/HugeCTR/src/regularizers/l1_regularizer.cu
  ${HugeCTR_Path}/HugeCTR/src/regularizers/l2_regularizer.cu
  ${HugeCTR_Path}/HugeCTR/src/regularizers/no_regularizer.cu
  ${HugeCTR_Path}/HugeCTR/src/hashtable/nv_hashtable.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/sync_all_gpus_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/init_embedding_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/forward_per_gpu_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/forward_scale_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/forward_reorder_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/forward_mapping_per_gpu_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/forward_fuse_per_gpu_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/store_slot_id_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/backward_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/backward_reorder_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/backward_fuse_per_gpu_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/update_params_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/get_update_params_results_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/reduce_scatter_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/all_reduce_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/all_gather_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/all2all_init_forward_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/all2all_init_backward_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/all2all_forward_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/all2all_backward_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/all2all_exec_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/get_forward_results_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/get_backward_results_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/utils_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/opt_states_functor.cu
  ${HugeCTR_Path}/HugeCTR/src/diagnose.cu
  ${HugeCTR_Path}/HugeCTR/src/utils.cu
)

FILE(GLOB  hugectr_partial_embedding
  ${HugeCTR_Path}/HugeCTR/src/embeddings/distributed_slot_sparse_embedding_hash.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/localized_slot_sparse_embedding_hash.cu
  ${HugeCTR_Path}/HugeCTR/src/embeddings/localized_slot_sparse_embedding_one_hot.cu
)

FILE(GLOB_RECURSE PLUGIN_SRC
${PROJECT_SOURCE_DIR}/cc/ops/v1/*.cc
${PROJECT_SOURCE_DIR}/cc/kernels/v1/*.cc
${PROJECT_SOURCE_DIR}/cc/kernels/v1/*.cu
)

FIND_PACKAGE(MPI)

ADD_LIBRARY(partial_embedding SHARED ${hugectr_partial_embedding})
target_compile_features(partial_embedding PRIVATE cxx_std_17)

ADD_LIBRARY(embedding_plugin SHARED ${PLUGIN_SRC} ${hugectr_src})
TARGET_LINK_LIBRARIES(embedding_plugin PUBLIC ${TF_LINK_FLAGS})

IF(MPI_FOUND)
  TARGET_LINK_LIBRARIES(partial_embedding PUBLIC stdc++fs cublas curand cudnn nccl nvToolsExt cusparse ${CMAKE_THREAD_LIBS_INIT} ${MPI_CXX_LIBRARIES})
  TARGET_LINK_LIBRARIES(embedding_plugin PUBLIC partial_embedding cublas curand cudnn nccl nvToolsExt cusparse ${CMAKE_THREAD_LIBS_INIT} ${MPI_CXX_LIBRARIES})
  MESSAGE(STATUS "${MPI_CXX_LIBRARIES}")
ELSE()
  TARGET_LINK_LIBRARIES(partial_embedding PUBLIC stdc++fs cublas curand cudnn nccl nvToolsExt cusparse ${CMAKE_THREAD_LIBS_INIT})
  TARGET_LINK_LIBRARIES(embedding_plugin PUBLIC partial_embedding cublas curand cudnn nccl nvToolsExt cusparse ${CMAKE_THREAD_LIBS_INIT})
ENDIF()

SET_TARGET_PROPERTIES(partial_embedding PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
SET_TARGET_PROPERTIES(partial_embedding PROPERTIES CUDA_ARCHITECTURES OFF)

SET_TARGET_PROPERTIES(embedding_plugin PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
SET_TARGET_PROPERTIES(embedding_plugin PROPERTIES CUDA_ARCHITECTURES OFF)


# ----------------------- V2 ----------------------------- #
file(GLOB_RECURSE PLUGIN_SRC_V2
${PROJECT_SOURCE_DIR}/cc/ops/v2/*.cc
${PROJECT_SOURCE_DIR}/cc/kernels/v2/*.cc
${PROJECT_SOURCE_DIR}/cc/kernels/v1/cuda_utils.cu
)

option(PLUGIN_NVTX "add nvtx on plugin ops" OFF)
if (PLUGIN_NVTX)
  MESSAGE("-- Add nvtx on plugin ops")
  ADD_DEFINITIONS(-DPLUGIN_NVTX)
endif()

ADD_LIBRARY(embedding_plugin_v2 SHARED ${PLUGIN_SRC_V2} ${hugectr_src})
TARGET_LINK_LIBRARIES(embedding_plugin_v2 PUBLIC ${TF_LINK_FLAGS})

IF(MPI_FOUND)
  TARGET_LINK_LIBRARIES(embedding_plugin_v2 PUBLIC partial_embedding cublas curand cudnn nccl nvToolsExt cusparse ${CMAKE_THREAD_LIBS_INIT} ${MPI_CXX_LIBRARIES})
  MESSAGE(STATUS "${MPI_CXX_LIBRARIES}")
ELSE()
  TARGET_LINK_LIBRARIES(embedding_plugin_v2 PUBLIC partial_embedding cublas curand cudnn nccl nvToolsExt cusparse ${CMAKE_THREAD_LIBS_INIT})
ENDIF()

SET_TARGET_PROPERTIES(embedding_plugin_v2 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
SET_TARGET_PROPERTIES(embedding_plugin_v2 PROPERTIES CUDA_ARCHITECTURES OFF)

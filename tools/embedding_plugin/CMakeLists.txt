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
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

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
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/cpu_resource.cpp
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/gpu_resource.cpp
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/resource_manager.cpp
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/data_simulator.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/optimizers/adam_optimizer.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/optimizers/momentum_sgd_optimizer.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/optimizers/nesterov_optimizer.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/optimizers/sgd_optimizer.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/optimizer.cpp
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/regularizer.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/regularizers/l1_regularizer.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/regularizers/l2_regularizer.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/regularizers/no_regularizer.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/hashtable/nv_hashtable.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/sync_all_gpus_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/init_embedding_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/forward_per_gpu_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/forward_scale_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/forward_reorder_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/forward_mapping_per_gpu_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/forward_fuse_per_gpu_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/store_slot_id_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/backward_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/backward_reorder_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/backward_fuse_per_gpu_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/update_params_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/get_update_params_results_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/reduce_scatter_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/all_reduce_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/all_gather_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/all2all_init_forward_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/all2all_init_backward_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/all2all_forward_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/all2all_backward_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/all2all_exec_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/get_forward_results_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/get_backward_results_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/utils_functor.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/distributed_slot_sparse_embedding_hash.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/localized_slot_sparse_embedding_hash.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/embeddings/localized_slot_sparse_embedding_one_hot.cu
  ${PROJECT_SOURCE_DIR}/HugeCTR/src/diagnose.cu
)

FILE(GLOB_RECURSE PLUGIN_SRC
${PROJECT_SOURCE_DIR}/tools/embedding_plugin/cc/ops/*.cc
${PROJECT_SOURCE_DIR}/tools/embedding_plugin/cc/kernels/*.cc
${PROJECT_SOURCE_DIR}/tools/embedding_plugin/cc/kernels/*.cu
)

FIND_PACKAGE(MPI)

ADD_LIBRARY(embedding_plugin SHARED ${PLUGIN_SRC} ${hugectr_src})
TARGET_LINK_LIBRARIES(embedding_plugin PUBLIC ${TF_LINK_FLAGS})

IF(MPI_FOUND)
  TARGET_LINK_LIBRARIES(embedding_plugin PUBLIC cublas curand cudnn nccl nvToolsExt cudf rmm cusparse ${CMAKE_THREAD_LIBS_INIT} ${MPI_CXX_LIBRARIES})
  MESSAGE(STATUS "${MPI_CXX_LIBRARIES}")
ELSE()
  TARGET_LINK_LIBRARIES(embedding_plugin PUBLIC cublas curand cudnn nccl nvToolsExt cudf rmm cusparse ${CMAKE_THREAD_LIBS_INIT})
ENDIF()
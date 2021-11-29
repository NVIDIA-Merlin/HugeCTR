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

cmake_minimum_required(VERSION 3.8)
include_directories(/opt/conda/include)
set(DB_LIB_PATHS "/usr/local/lib" CACHE PATH "Paths to Hiredis/rocksdb lib")

file(GLOB huge_ctr_src
  ../cpu_resource.cpp
  ../gpu_resource.cpp
  ../resource_manager.cpp
  ../resource_managers/resource_manager_core.cpp
  ../data_simulator.cu
  ../data_reader.cpp
  ../graph_wrapper.cpp
  ../layer.cpp
  ../layers/batch_norm_layer.cu
  ../layers/cast_layer.cu
  ../layers/concat_layer.cu
  ../layers/dropout_layer.cu
  ../layers/elu_layer.cu
  ../layers/fully_connected_layer.cu
  ../layers/fully_connected_layer_half.cu
  ../layers/fused_fully_connected_layer.cu
  ../layers/fused_relu_bias_fully_connected_layer.cu
  ../layers/interaction_layer.cu
  ../layers/relu_layer.cu
  ../layers/reshape_layer.cu
  ../layers/sigmoid_layer.cu
  ../layers/slice_layer.cu
  ../layers/fm_order2_layer.cu
  ../layers/weight_multiply_layer.cu
  ../layers/multi_cross_layer.cu
  ../layers/add_layer.cu
  ../layers/reduce_sum_layer.cu
  ../layers/dot_product_layer.cu
  ../layers/elementwise_multiply_layer.cu
  ../layers/gru_layer.cu
  ../layers/matrix_multiply_layer.cu
  ../layers/prelu_dice_layer.cu
  ../layers/softmax_layer.cu
  ../layers/scale_layer.cu
  ../layers/fused_reshape_concat_general_layer.cu
  ../layers/fused_reshape_concat_layer.cu
  ../layers/sub_layer.cu
  ../layers/gather_layer.cu
  ../layers/reduce_mean_layer.cu
  ../loss.cu
  ../network.cu
  ../network.cpp
  embedding_feature_combiner.cu
  hugectrmodel.cpp
  embedding_cache.cpp
  session_inference.cpp
  embedding_cache.cu
  embedding_interface.cpp
  parameter_server.cpp
  inference_utilis.cpp
  redis_backend.cpp
  rocksdb_backend.cpp
  sha1.cpp
  utils.cpp
  local_memory_backend.cpp
  database_backend.cpp
  unique_op/unique_op.cu
  ../data_readers/metadata.cpp
  ../gpu_learning_rate_scheduler.cu
  ../metrics.cu
  ../optimizers/*.cu
  ../optimizer.cpp
  ../thread_pool.cpp
  ../regularizer.cu
  ../regularizers/l1_regularizer.cu
  ../regularizers/l2_regularizer.cu
  ../regularizers/no_regularizer.cu
  ../parsers/solver_parser.cpp
  ../parsers/learning_rate_scheduler_parser.cpp
  ../parsers/create_optimizer.cpp
  ../parsers/create_network.cpp
  ../parsers/inference_parser.cpp
  ../diagnose.cu
  ../utils.cu
  ../base/debug/logger.cpp
  ../base/debug/cuda_debugging.cu
)

set(CMAKE_CXX_STANDARD 17)
add_library(hugectr_inference SHARED ${huge_ctr_src})

target_link_libraries(hugectr_inference PUBLIC cudart cublas curand cudnn nccl cudf stdc++fs ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(hugectr_inference PUBLIC gpu_cache)

target_link_libraries(hugectr_inference PRIVATE nlohmann_json::nlohmann_json nvToolsExt)

target_link_libraries(
  hugectr_inference
  PUBLIC
    ${DB_LIB_PATHS}/libhiredis.so librocksdb.so libredis++.so  # from Hugectr
)

target_compile_features(hugectr_inference PUBLIC cxx_std_17)

set_target_properties(hugectr_inference PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(hugectr_inference PROPERTIES CUDA_ARCHITECTURES OFF)

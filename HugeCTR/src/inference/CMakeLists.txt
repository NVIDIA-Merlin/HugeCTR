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

cmake_minimum_required(VERSION 3.8)
include_directories(/opt/conda/include)

file(GLOB huge_ctr_src
  ../cpu_resource.cpp
  ../gpu_resource.cpp
  ../resource_manager.cpp
  ../data_simulator.cu
  ../data_reader.cpp
  ../layer.cpp
  ../layers/batch_norm_layer.cu
  ../layers/cast_layer.cu
  ../layers/concat_layer.cu
  ../layers/dropout_layer.cu
  ../layers/dropout_cudnn_layer.cu
  ../layers/elu_layer.cu
  ../layers/fully_connected_layer.cu
  ../layers/fully_connected_layer_half.cu
  ../layers/fused_fully_connected_layer.cu
  ../layers/interaction_layer.cu
  ../layers/relu_layer.cu
  ../layers/reshape_layer.cu
  ../layers/sigmoid_layer.cu
  ../layers/slice_layer.cu
  ../layers/fm_order2_layer.cu
  ../layers/multiply_layer.cu
  ../layers/multi_cross_layer.cu
  ../layers/add_layer.cu
  ../layers/reduce_sum_layer.cu
  ../layers/dot_product_layer.cu
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
  gpu_cache/nv_gpu_cache.cu
  ../data_readers/metadata.cpp
  ../metrics.cu
  ../optimizers/adam_optimizer.cu
  ../optimizers/momentum_sgd_optimizer.cu
  ../optimizers/nesterov_optimizer.cu
  ../optimizers/sgd_optimizer.cu
  ../optimizer.cpp
  ../regularizer.cu
  ../regularizers/l1_regularizer.cu
  ../regularizers/l2_regularizer.cu
  ../regularizers/no_regularizer.cu
  ../parser.cpp
  ../parsers/solver_parser.cpp
  ../parsers/learning_rate_scheduler_parser.cpp
  ../parsers/create_optimizer.cpp
  ../parsers/create_network.cpp
#   ../parsers/create_embedding.cpp
#   ../parsers/create_datareader.cpp
  ../parsers/inference_parser.cpp
  ../session.cpp
  ../plan_parser.cpp 
  ../model_oversubscriber/distributed_parameter_server_delegate.cpp
  ../model_oversubscriber/model_oversubscriber_impl.cpp
  ../model_oversubscriber/parameter_server.cpp
  ../model_oversubscriber/parameter_server_manager.cpp
  ../diagnose.cu
)

set(CMAKE_CXX_STANDARD 14)
add_library(hugectr_inference SHARED ${huge_ctr_src})

target_link_libraries(hugectr_inference PUBLIC cublas curand cudnn nccl ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(hugectr_inference PRIVATE nlohmann_json::nlohmann_json)

target_compile_features(hugectr_inference PUBLIC cxx_std_14)

set_target_properties(hugectr_inference PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

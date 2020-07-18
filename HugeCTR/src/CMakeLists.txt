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

file(GLOB huge_ctr_src
  gpu_resource.cpp
  data_parser.cpp
  data_reader.cpp
  layer.cpp
  layers/batch_norm_layer.cu
  layers/cast_layer.cu
  layers/concat_layer.cu
  layers/dropout_layer.cu
  layers/elu_layer.cu
  layers/fully_connected_layer.cu
  layers/fully_connected_layer_half.cu
  layers/fused_fully_connected_layer.cu
  layers/interaction_layer.cu
  layers/relu_layer.cu
  layers/reshape_layer.cu
  layers/slice_layer.cu
  layers/fm_order2_layer.cu
  layers/multiply_layer.cu
  layers/multi_cross_layer.cu
  layers/add_layer.cu
  layers/reduce_sum_layer.cu
  loss.cu
  network.cu
  network.cpp
  metrics.cu
  optimizers/adam_optimizer.cu
  optimizers/momentum_sgd.cu
  optimizers/nesterov_optimizer.cu
  optimizers/sgd_optimizer.cu
  regularizer.cu
  regularizers/l1_regularizer.cu
  regularizers/l2_regularizer.cu
  regularizers/no_regularizer.cu
  parser.cpp
  parsers/solver_parser.cpp
  parsers/learning_rate_scheduler_parser.cpp
  session.cpp
  plan_parser.cpp 
  embedding_creator.cu
  data_collector.cu
)


add_library(huge_ctr_static STATIC ${huge_ctr_src})

if(MPI_FOUND)
  target_link_libraries(huge_ctr_static PUBLIC cublas curand cudnn nccl nvToolsExt ${CMAKE_THREAD_LIBS_INIT} ${MPI_CXX_LIBRARIES} hwloc ucp ucs ucm uct)
  message(STATUS "${MPI_CXX_LIBRARIES}")
else()
  target_link_libraries(huge_ctr_static PUBLIC cublas curand cudnn nccl nvToolsExt ${CMAKE_THREAD_LIBS_INIT})
endif()

target_link_libraries(huge_ctr_static PRIVATE nlohmann_json::nlohmann_json)

target_compile_features(huge_ctr_static PUBLIC cxx_std_14)

set_target_properties(huge_ctr_static PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(huge_ctr main.cpp)
target_link_libraries(huge_ctr PUBLIC huge_ctr_static)
target_compile_features(huge_ctr PUBLIC cxx_std_14)

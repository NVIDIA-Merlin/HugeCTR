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


file(GLOB cpu_inference_src
  layers/add_layer_cpu.cpp
  layers/batch_norm_layer_cpu.cpp
  layers/cast_layer_cpu.cpp
  layers/concat_layer_cpu.cpp
  layers/dropout_layer_cpu.cpp
  layers/elu_layer_cpu.cpp
  layers/fm_order2_layer_cpu.cpp
  layers/fully_connected_layer_cpu.cpp
  layers/fully_connected_layer_half_cpu.cpp
  layers/fused_fully_connected_layer_cpu.cpp
  layers/interaction_layer_cpu.cpp
  layers/multi_cross_layer_cpu.cpp
  layers/reduce_sum_layer_cpu.cpp
  layers/relu_layer_cpu.cpp
  layers/reshape_layer_cpu.cpp
  layers/sigmoid_layer_cpu.cpp
  layers/slice_layer_cpu.cpp
  layers/weight_multiply_layer_cpu.cpp
  network_cpu.cpp
  embedding_feature_combiner_cpu.cpp
  create_network_cpu.cpp
  create_embedding_cpu.cpp
  create_pipeline_cpu.cpp
  session_inference_cpu.cpp
)

set(CMAKE_CXX_STANDARD 17)

add_library(cpu_inference_shared SHARED ${cpu_inference_src})

target_link_libraries(cpu_inference_shared PUBLIC cudart cublas curand cudnn nccl ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(cpu_inference_shared PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(cpu_inference_shared PUBLIC hugectr_inference)

target_compile_features(cpu_inference_shared PUBLIC cxx_std_17)

set_target_properties(cpu_inference_shared PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(cpu_inference_shared PROPERTIES CUDA_ARCHITECTURES OFF)

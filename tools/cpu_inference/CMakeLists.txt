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


file(GLOB cpu_inference_src
  layers/src/add_layer_cpu.cpp
  layers/src/batch_norm_layer_cpu.cpp
  layers/src/cast_layer_cpu.cpp
  layers/src/concat_layer_cpu.cpp
  layers/src/dot_product_layer_cpu.cpp
  layers/src/dropout_layer_cpu.cpp
  layers/src/elu_layer_cpu.cpp
  layers/src/fm_order2_layer_cpu.cpp
  layers/src/fully_connected_layer_cpu.cpp
  layers/src/fully_connected_layer_half_cpu.cpp
  layers/src/fused_fully_connected_layer_cpu.cpp
  layers/src/interaction_layer_cpu.cpp
  layers/src/multi_cross_layer_cpu.cpp
  layers/src/reduce_sum_layer_cpu.cpp
  layers/src/relu_layer_cpu.cpp
  layers/src/reshape_layer_cpu.cpp
  layers/src/sigmoid_layer_cpu.cpp
  layers/src/slice_layer_cpu.cpp
  layers/src/weight_multiply_layer_cpu.cpp
  network_cpu.cpp
  embedding_feature_combiner_cpu.cpp
  create_network_cpu.cpp
  create_embedding_cpu.cpp
  create_pipeline_cpu.cpp
  session_inference_cpu.cpp
)

set(CMAKE_CXX_STANDARD 14)

add_library(cpu_inference_shared SHARED ${cpu_inference_src})

target_link_libraries(cpu_inference_shared PUBLIC cudart cublas curand cudnn nccl ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(cpu_inference_shared PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(cpu_inference_shared PUBLIC hugectr_inference)

target_compile_features(cpu_inference_shared PUBLIC cxx_std_14)

set_target_properties(cpu_inference_shared PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(cpu_inference_shared PROPERTIES CUDA_ARCHITECTURES OFF)

# 
# Copyright (c) 2019, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.8)

file(GLOB huge_ctr_src
  data_parser.cpp
  data_reader.cpp
  layer.cpp
  layers/batch_norm_layer.cu
  layers/concat_layer.cu
  layers/elu_layer.cu
  layers/fully_connected_layer.cu
  layers/relu_layer.cu
  loss.cu
  network.cpp
  optimizers/adam_optimizer.cu
  optimizers/momentum_sgd.cu
  optimizers/nesterov_optimizer.cu
  parser.cpp
  session.cpp
  embedding_creator.cu
  data_collector_ex.cu
)


add_library(huge_ctr_static STATIC ${huge_ctr_src})

if(MPI_FOUND)
  target_link_libraries(huge_ctr_static PUBLIC cublas cudnn nccl nvToolsExt ${CMAKE_THREAD_LIBS_INIT} ${MPI_CXX_LIBRARIES})
  message(STATUS "${MPI_CXX_LIBRARIES}")
else()
  target_link_libraries(huge_ctr_static PUBLIC cublas cudnn nccl nvToolsExt ${CMAKE_THREAD_LIBS_INIT})
endif()

target_link_libraries(huge_ctr_static PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(huge_ctr_static PUBLIC cxx_std_11)

add_executable(huge_ctr main.cpp)
target_link_libraries(huge_ctr PUBLIC huge_ctr_static)
target_compile_features(huge_ctr PUBLIC cxx_std_11)

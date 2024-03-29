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

cmake_minimum_required(VERSION 3.20)
file(GLOB model_generator_src
  model_generator/generate_embedding.cpp
)

file(GLOB triton_request_generator_src
  request_generator/generate_triton_request.cpp
)

add_executable(model_generator ${model_generator_src})
target_compile_features(model_generator PUBLIC cxx_std_17)
target_link_libraries(model_generator PUBLIC huge_ctr_shared)

add_executable(request_generator ${triton_request_generator_src})
target_compile_features(request_generator PUBLIC cxx_std_17)
target_link_libraries(request_generator PUBLIC huge_ctr_shared)

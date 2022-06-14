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


cmake_minimum_required(VERSION 3.17)
find_package(CUDAToolkit)
include_directories(/opt/conda/include)

file(GLOB embedding_src
*.cpp
*.cu
operators/*.cpp
operators/*.cu
)

add_library(embedding SHARED ${embedding_src})

set_target_properties(embedding PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(embedding PUBLIC core)
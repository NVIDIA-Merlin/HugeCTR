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

file(GLOB core_src 
*.cpp
../src/base/debug/logger.cpp # use link instead in future
)

option(ENABLE_HCTR "Enable HugeCTR backend" OFF)
if(ENABLE_HCTR)
  file(GLOB hctr_impl ./hctr_impl/*.cpp)
  set(core_src ${core_src} ${hctr_impl})
endif()

option(ENABLE_TF "Enable TF backend" OFF)
if(ENABLE_TF)
  include_directories(/usr/local/lib/python3.8/dist-packages/tensorflow/include)
  link_directories(/usr/local/lib/python3.8/dist-packages/tensorflow)
  file(GLOB tf_impl ./tf_impl/*.cpp)
  set(core_src ${core_src} ${tf_impl})
endif()

message(STATUS "foo include dir: ${core_src}")
add_library(core SHARED ${core_src})

target_link_libraries(core PUBLIC cudart nccl)
if (ENABLE_MULTINODES)
    target_link_libraries(core PUBLIC ${MPI_CXX_LIBRARIES} hwloc ucp ucs ucm)
endif()
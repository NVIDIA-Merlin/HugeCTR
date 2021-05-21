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
file(GLOB_RECURSE huge_ctr_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)
list(FILTER huge_ctr_src EXCLUDE REGEX ".*main.cpp")

add_library(huge_ctr_static STATIC ${huge_ctr_src})


if(MPI_FOUND)
  target_link_libraries(huge_ctr_static PUBLIC cublas cublasLt curand cudnn nccl nvToolsExt ${CMAKE_THREAD_LIBS_INIT} ${MPI_CXX_LIBRARIES} hwloc ucp ucs ucm uct numa ibverbs gdrapi)
  message(STATUS "${MPI_CXX_LIBRARIES}")
else()
  target_link_libraries(huge_ctr_static PUBLIC cublas cublasLt curand cudnn nccl nvToolsExt ${CMAKE_THREAD_LIBS_INIT} numa)
endif()

if(NOT DISABLE_CUDF)
  target_link_libraries(huge_ctr_static PUBLIC cudf)
endif()

if (ENABLE_MULTINODES)
  if (SHARP_FOUND)
    target_link_libraries(huge_ctr_static PRIVATE sharp sharp_coll)
  else()
    message(FATAL_ERROR "Multi-node enabled but SHARP not found")
  endif()
endif()

target_link_libraries(huge_ctr_static PRIVATE CUDA::nvml nlohmann_json::nlohmann_json aio numa)

target_compile_features(huge_ctr_static PUBLIC cxx_std_14)

set_target_properties(huge_ctr_static PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(huge_ctr main.cpp)
target_link_libraries(huge_ctr PUBLIC huge_ctr_static)
target_compile_features(huge_ctr PUBLIC cxx_std_14)

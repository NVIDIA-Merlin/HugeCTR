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
#


cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

file(GLOB core23_src 
allocator_factory.cpp
allocator_params.cpp
buffer.cpp
buffer_client.cpp
buffer_channel.cpp
buffer_channel_helpers.cpp
buffer_factory.cpp
device.cpp
device_guard.cpp
device_type.cpp
data_type.cpp
offsetted_buffer.cpp
low_level_primitives.cpp
low_level_primitives.cu
mpi_init_service.cpp
details/simple_cuda_allocator.cpp
details/managed_cuda_allocator.cpp
details/low_level_cuda_allocator.cpp
details/pool_cuda_allocator.cpp
details/pinned_host_allocator.cpp
details/new_delete_allocator.cpp
details/unitary_buffer.cpp
details/confederal_buffer.cpp
details/tensor_impl.cpp
details/tensor_helpers.cpp
details/buffer_factory_impl.cpp
details/host_launch_helpers.cpp
tensor.cpp
tensor_operations.cpp
shape.cpp
../src/base/debug/logger.cpp # use link instead in future
)

add_library(hugectr_core23 SHARED ${core23_src})

set_target_properties(hugectr_core23 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(hugectr_core23 PUBLIC cuda cudart curand)
if (ENABLE_MULTINODES)
    target_link_libraries(hugectr_core23 PUBLIC ${MPI_CXX_LIBRARIES} hwloc ucp ucs ucm)
endif()

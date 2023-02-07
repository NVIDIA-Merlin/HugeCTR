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
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.17)

# TODO: consider using benchmark::benchmark
function(configureBenchmark executableName)
  add_executable(${executableName} ${ARGN})
  target_compile_features(${executableName} PUBLIC cxx_std_17)
  target_link_libraries(${executableName} PUBLIC hugectr_core23)
  set_target_properties(${executableName} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                                     CUDA_ARCHITECTURES OFF
                                                     POSITION_INDEPENDENT_CODE ON)
endfunction(configureBenchmark)


configureBenchmark(random_allocation_bench random_allocations.cpp)
configureBenchmark(tensor_performance_bench tensor_performance.cu)
configureBenchmark(tensor_container_performance_bench tensor_container_performance.cu)
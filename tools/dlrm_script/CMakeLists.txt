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
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.8)
file(GLOB dlrm_raw_src
  dlrm_raw.cu
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

set(CONDA_PREFIX /opt/conda)

include_directories(${CONDA_PREFIX}/include)
include_directories(${PROJECT_SOURCE_DIR}/tools/dlrm_script)
include_directories(${PROJECT_SOURCE_DIR}/third_party/cub)
include_directories(${CONDA_PREFIX}/include/libcudf/libcudacxx)

link_directories(${CONDA_PREFIX}/lib)

add_executable(dlrm_raw ${dlrm_raw_src})

if(MPI_FOUND)
  target_link_libraries(dlrm_raw PUBLIC cudart cudf boost_filesystem ${MPI_CXX_LIBRARIES})
else()
  target_link_libraries(dlrm_raw PUBLIC cudart cudf boost_filesystem)
endif()
set_target_properties(dlrm_raw PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_features(dlrm_raw PUBLIC cxx_std_14)




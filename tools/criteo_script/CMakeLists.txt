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
file(GLOB criteo2hugectr_src
  criteo2hugectr.cpp
)

add_executable(criteo2hugectr ${criteo2hugectr_src})
target_compile_features(criteo2hugectr PUBLIC cxx_std_14)
if(MPI_FOUND)
  target_link_libraries(criteo2hugectr PUBLIC ${MPI_CXX_LIBRARIES})
endif()



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
add_subdirectory(core23_layer_test)
add_subdirectory(data_distributor)
add_subdirectory(data_reader)
add_subdirectory(optimizer)
add_subdirectory(loss)
add_subdirectory(metrics)
add_subdirectory(network)
add_subdirectory(regularizers)
add_subdirectory(device_map)
if (NOT DISABLE_CUDF)
  add_subdirectory(embedding)
endif()
add_subdirectory(checker)
add_subdirectory(prims)
add_subdirectory(misc)
add_subdirectory(embedding_collection)
add_subdirectory(io)
add_subdirectory(communication)
add_subdirectory(pipeline)
add_subdirectory(core23)
add_subdirectory(dense_layer)

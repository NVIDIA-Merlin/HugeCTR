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
find_package(CUDAToolkit)
set(DB_LIB_PATHS "/usr/local/lib" CACHE PATH "Paths to Hiredis/RocksDB lib")

include(FetchContent)

FetchContent_Declare(
  pybind11_sources
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        master
)
FetchContent_GetProperties(pybind11_sources)

if(NOT pybind11_sources_POPULATED)
  FetchContent_Populate(pybind11_sources)
  add_subdirectory(
    ${pybind11_sources_SOURCE_DIR}
    ${pybind11_sources_BINARY_DIR}
    )
endif()

file(
  GLOB_RECURSE huge_ctr_src 
  RELATIVE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    *.cpp
    *.cu
    ../embedding_storage/*.cpp
    ../embedding_storage/*.cu
    ../../third_party/dynamic_embedding_table/*.cpp
    ../../third_party/dynamic_embedding_table/*.cu
    )

list(REMOVE_ITEM huge_ctr_src "pybind/module_main.cpp")
list(REMOVE_ITEM huge_ctr_src "inference_benchmark/metrics.cpp")

if(DISABLE_CUDF)
  list(REMOVE_ITEM huge_ctr_src "data_readers/file_source_parquet.cpp")
  list(REMOVE_ITEM huge_ctr_src "data_readers/metadata.cpp")
  list(REMOVE_ITEM huge_ctr_src "data_readers/parquet_data_reader_worker.cpp")
  list(REMOVE_ITEM huge_ctr_src "data_readers/row_group_reading_thread.cpp")
  list(REMOVE_ITEM huge_ctr_src "data_readers/dataframe_container.cu")
  list(REMOVE_ITEM huge_ctr_src "data_readers/parquet_data_converter.cu")
endif()

add_library(huge_ctr_shared SHARED ${huge_ctr_src})
target_link_libraries(huge_ctr_shared PUBLIC hugectr_core23 embedding)
target_link_libraries(huge_ctr_shared PUBLIC CUDA::cuda_driver ${CUDART_LIB} CUDA::cublasLt CUDA::cublas CUDA::curand CUDA::nvml CUDA::nvToolsExt cudnn nccl)
target_link_libraries(huge_ctr_shared PUBLIC ${CMAKE_THREAD_LIBS_INIT} numa stdc++fs tbb rdkafka)
target_link_libraries(huge_ctr_shared PRIVATE aio)
target_link_libraries(huge_ctr_shared PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(huge_ctr_shared PUBLIC gpu_cache)

if(ENABLE_HDFS)
  target_link_libraries(huge_ctr_shared PUBLIC ${DB_LIB_PATHS}/libhdfs.so)
endif()

if(ENABLE_S3)
  target_link_libraries(huge_ctr_shared PUBLIC ${DB_LIB_PATHS}/libaws-cpp-sdk-core.so ${DB_LIB_PATHS}/libaws-cpp-sdk-s3.so)
endif()

if(ENABLE_GCS)
  target_link_libraries(huge_ctr_shared PUBLIC google_cloud_cpp_storage)
endif()

if (ENABLE_MULTINODES)
  target_link_libraries(huge_ctr_shared PUBLIC ${MPI_CXX_LIBRARIES} hwloc ucp ucs ucm uct ibverbs gdrapi stdc++fs)
  if (SHARP_FOUND)
    target_link_libraries(huge_ctr_shared PRIVATE sharp sharp_coll)
  else()
    message(FATAL_ERROR "Multi-node enabled but SHARP not found")
  endif()
endif()

if(NOT DISABLE_CUDF)
  target_link_libraries(huge_ctr_shared PUBLIC cudf)
  if(Parquet_FOUND)
    target_link_libraries(huge_ctr_shared PUBLIC parquet)
  endif()
endif()

target_compile_features(huge_ctr_shared PUBLIC cxx_std_17 cuda_std_17)

add_library(hugectr MODULE pybind/module_main.cpp)
target_link_libraries(hugectr PUBLIC pybind11::module python$ENV{PYTHON_VERSION} huge_ctr_shared)
set_target_properties(hugectr PROPERTIES PREFIX "")

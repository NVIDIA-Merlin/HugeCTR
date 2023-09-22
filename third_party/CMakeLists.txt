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

cmake_minimum_required(VERSION 3.20)
message(STATUS "Configuring third_party dependencies...")

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

# NOTE: Confine individual dependencies in a function scope to avoid unintentionally
#       leaking genericly named variables (e.g.: WITH_SSL).

function(build_gtest)
  message("=== Dependency: Google Test ===================================================")
  set(BUILD_GMOCK ON)
  add_subdirectory(googletest)
endfunction()
build_gtest()

function(build_json)
  message("=== Dependency: NLohmann / JSON ===============================================")
  set(JSON_BuildTests OFF)
  if(ENABLE_GCS)
    set(JSON_Install ON)
  endif()
  add_subdirectory(json)
endfunction()
build_json()

function(build_redis)
  list(FIND HCTR_FEATURES "redis" HCTR_FEATURE_ENABLED)
  if(${HCTR_FEATURE_ENABLED} EQUAL -1)
    return()
  endif()

  message("=== Configure Dependency: Hiredis =============================================")
  set(ENABLE_SSL ON)
  set(DISABLE_TESTS ON)
  add_subdirectory(hiredis)
  
  set(stub_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/pkg")
  file(WRITE "${stub_dir}/hiredis-config.cmake" "")
  file(WRITE "${stub_dir}/hiredis_ssl-config.cmake" "")
  
  set(hiredis_DIR ${stub_dir})
  set(hiredis_LIBRARIES hiredis::hiredis)
  set(hiredis_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/_deps")
  if(NOT TARGET hiredis::hiredis)
    add_library(hiredis::hiredis ALIAS hiredis)
  endif()
  
  set(hiredis_ssl_DIR ${stub_dir})
  set(hiredis_ssl_LIBRARIES hiredis::hiredis_ssl)
  set(hiredis_ssl_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/_deps")
  if(NOT TARGET hiredis::hiredis_ssl)
    add_library(hiredis::hiredis_ssl ALIAS hiredis_ssl)
  endif()

  message("=== Configure Dependency: Redis++ =============================================")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  set(REDIS_PLUS_PLUS_BUILD_STATIC OFF)
  set(REDIS_PLUS_PLUS_BUILD_TEST OFF)
  set(REDIS_PLUS_PLUS_USE_TLS ON)
  add_subdirectory(redis_pp)
endfunction()
build_redis()

function(build_rocksdb)
  list(FIND HCTR_FEATURES "rocks_db" HCTR_FEATURE_ENABLED)
  if(${HCTR_FEATURE_ENABLED} EQUAL -1)
    return()
  endif()

  message("=== Configure Dependency: RocksDB =============================================")
  set(PORTABLE ON)
  set(WITH_TESTS OFF)
  set(WITH_BENCHMARK_TOOLS OFF)
  set(WITH_TOOLS OFF)
  add_subdirectory(rocksdb)
endfunction()
build_rocksdb()

function(build_librdkafka)
  message("=== Dependency: librdkafka ====================================================")
  set(RDKAFKA_BUILD_EXAMPLES OFF)
  set(RDKAFKA_BUILD_TESTS OFF)
  add_subdirectory(librdkafka)
endfunction()
build_librdkafka()

function(build_gcs)
  message("=== Optional dependency: Abseil C++ ===========================================")
  find_package(absl QUIET)
  if(${absl_FOUND})
    message("Abseil is already present.")
  else()
    message("Couldn't find abseil. Attempting to download, build and install it now.")
    FetchContent_Declare(abseil-cpp
      DOWNLOAD_COMMAND git clone --branch 20230125.3 --depth 1 https://github.com/abseil/abseil-cpp.git ${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src
    )
    FetchContent_Populate(abseil-cpp)
    execute_process(WORKING_DIRECTORY ${abseil-cpp_BINARY_DIR}
      COMMAND cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DABSL_BUILD_TESTING=OFF -DABSL_PROPAGATE_CXX_STD=ON ${abseil-cpp_SOURCE_DIR}
    )
    execute_process(WORKING_DIRECTORY ${abseil-cpp_BINARY_DIR}
      COMMAND make -j install
    )
    find_package(absl REQUIRED)
  endif()

  message("=== Optional dependency: CRC32C ===============================================")
  find_package(Crc32c QUIET)
  if(${Crc32c_FOUND})
    message("CRC32C is already present.")
  else()
    message("Couldn't find CRC32C. Attempting to download, build and install it now.")
    FetchContent_Declare(crc32c
      DOWNLOAD_COMMAND git clone --branch 1.1.2 --depth 1 https://github.com/google/crc32c ${CMAKE_BINARY_DIR}/_deps/crc32c-src
    )
    FetchContent_Populate(crc32c)
    execute_process(WORKING_DIRECTORY ${crc32c_BINARY_DIR}
      COMMAND cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCRC32C_BUILD_TESTS=OFF -DCRC32C_BUILD_BENCHMARKS=OFF -DCRC32C_USE_GLOG=OFF ${crc32c_SOURCE_DIR}
    )
    execute_process(WORKING_DIRECTORY ${crc32c_BINARY_DIR}
      COMMAND make -j install
    )
  endif()

  message("=== Optional dependency: Google Cloud Storage C++ Client Library ==============")
  FetchContent_Declare(google-cloud-cpp
    DOWNLOAD_COMMAND git clone --branch v2.12.0 --depth 1 https://github.com/googleapis/google-cloud-cpp.git ${CMAKE_BINARY_DIR}/_deps/google-cloud-cpp-src
  )
  FetchContent_Populate(google-cloud-cpp)
  
  # NLohmann / JSON is already available and the discovery here will fail.
  # Must check whenever `--branch` is adjusted in the `git clone` command above.
  execute_process(WORKING_DIRECTORY ${google-cloud-cpp_SOURCE_DIR}
    COMMAND sed -i /find_nlohmann_json\(\)/d cmake/IncludeNlohmannJson.cmake
  )

  set(CMAKE_CXX_FLAGS "-fPIC")
  set(BUILD_SHARED_LIBS ON)
  set(BUILD_TESTING OFF)
  set(GOOGLE_CLOUD_CPP_ENABLE_EXAMPLES OFF)
  set(GOOGLE_CLOUD_CPP_ENABLE "storage")
  set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/json/include")
  find_path(NLOHMANN_JSON_EXISTS "nlohmann/json.hpp" REQUIRED)
  add_subdirectory(${google-cloud-cpp_SOURCE_DIR} ${google-cloud-cpp_BINARY_DIR})
endfunction()
if (ENABLE_GCS)
  build_gcs()
endif()

message("=== Dependency configuration complete =========================================")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  # Set default installation to /usr/local
  set(CMAKE_INSTALL_PREFIX /usr/local CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
endif()



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
message(STATUS "Building third_party dependencies.")
set(CMAKE_CXX_STANDARD 17)
#install dependencies
#hiredis
add_subdirectory(hiredis)
set(stub_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/pkg")
file(WRITE "${stub_dir}/hiredis-config.cmake" "")
set(hiredis_DIR ${stub_dir})
set(hiredis_LIBRARIES hiredis::hiredis)
set(hiredis_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/_deps")
if(NOT TARGET hiredis::hiredis)
  add_library(hiredis::hiredis ALIAS hiredis)
endif()

#redis++
set(REDIS_PLUS_PLUS_BUILD_STATIC OFF CACHE BOOL "Build static library")
set(REDIS_PLUS_PLUS_BUILD_TEST OFF CACHE BOOL "Build test_redes++")
add_subdirectory(redis_pp)

#rocksdb
set(PORTABLE ON CACHE BOOL "build a portable binary")
set(WITH_TESTS OFF CACHE BOOL "build with tests")
set(WITH_BENCHMARK_TOOLS OFF CACHE BOOL "build with benchmark tools")
set(WITH_TOOLS OFF CACHE BOOL "build with tools")
add_subdirectory(rocksdb)

#librdkafka
set(RDKAFKA_BUILD_EXAMPLES OFF CACHE BOOL "rdkafka build examples")
set(RDKAFKA_BUILD_TESTS OFF CACHE BOOL "rdkafka build tests")
add_subdirectory(librdkafka)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
# Set default installation to /usr/local
  set(CMAKE_INSTALL_PREFIX /usr/local CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
endif()



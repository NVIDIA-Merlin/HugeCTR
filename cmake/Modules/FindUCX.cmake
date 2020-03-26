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

set(UCX_INC_PATHS
    /usr/include
    /usr/local/include
    $ENV{UCX_DIR}/include
    )

set(UCX_LIB_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    $ENV{UCX_DIR}/lib
    )

list(APPEND UCX_NAMES ucp libucp ucs libucs ucm libucm uct libuct)

#find_path(UCX_INCLUDE_DIR NAMES ucp.h PATHS ${UCX_INC_PATHS})
#find_library(UCX_LIBRARIES NAMES hwloc PATHS ${UCX_LIB_PATHS})
find_path(UCX_INCLUDE_DIR NAMES ucp/api/ucp.h HINTS ${UCX_INSTALL_DIR} PATH_SUFFIXES include)
find_library(UCX_LIBRARIES NAMES ${UCX_NAMES} HINTS ${UCX_INSTALL_DIR} PATH_SUFFIXES lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UCX DEFAULT_MSG UCX_INCLUDE_DIR UCX_LIBRARIES)

if (UCX_FOUND)
  message(STATUS "Found UCX    (include: ${UCX_INCLUDE_DIR}, library: ${UCX_LIBRARIES})")
  mark_as_advanced(UCX_INCLUDE_DIR UCX_LIBRARIES)
endif ()

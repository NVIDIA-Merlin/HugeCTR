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

set(HWLOC_INC_PATHS
    /usr/include
    /usr/local/include
    $ENV{HWLOC_DIR}/include
    $ENV{HWLOC}/include
    )

set(HWLOC_LIB_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    $ENV{HWLOC_DIR}/lib
    $ENV{HWLOC}/lib
    )

find_path(HWLOC_INCLUDE_DIR NAMES hwloc.h PATHS ${HWLOC_INC_PATHS})
find_library(HWLOC_LIBRARIES NAMES hwloc PATHS ${HWLOC_LIB_PATHS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HWLOC DEFAULT_MSG HWLOC_INCLUDE_DIR HWLOC_LIBRARIES)

if (HWLOC_FOUND)
  message(STATUS "Found HWLOC    (include: ${HWLOC_INCLUDE_DIR}, library: ${HWLOC_LIBRARIES})")
  mark_as_advanced(HWLOC_INCLUDE_DIR HWLOC_LIBRARIES)
endif ()

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

set(SHARP_INC_PATHS
  /opt/hpcx/sharp/include/sharp
  /usr/include
  /usr/local/include
  $ENV{SHARP_DIR}/include
  )

set (SHARP_LIB_PATHS
  /opt/hpcx/sharp/lib
  /lib
  /lib64
  /usr/lib
  /usr/lib64
  /usr/local/lib
  /usr/local/lib64
  $ENV{SHARP_DIR}/lib
  )

list(APPEND SHARP_NAMES sharp libsharp sharp_coll libsharp_coll)
find_path(SHARP_INCLUDE_DIR NAMES api/sharp.h PATHS ${SHARP_INC_PATHS})
find_library(SHARP_LIBRARIES NAMES ${SHARP_NAMES} PATHS ${SHARP_LIB_PATHS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SHARP DEFAULT_MSG SHARP_INCLUDE_DIR SHARP_LIBRARIES)

if (SHARP_FOUND)
  message(STATUS "Found SHARP    (include: ${SHARP_INCLUDE_DIR}, library: ${SHARP_LIBRARIES})")
  mark_as_advanced(SHARP_INCLUDE_DIR SHARP_LIBRARIES)
endif ()

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

set(CUDNN_INC_PATHS
    /usr/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    /usr/local/include
    $ENV{CUDNN_DIR}/include
    )

set(CUDNN_LIB_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    /usr/local/lib
    /usr/local/lib64
    $ENV{CUDNN_DIR}/lib64
    )

find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h PATHS ${CUDNN_INC_PATHS})
find_library(CUDNN_LIBRARIES NAMES cudnn PATHS ${CUDNN_LIB_PATHS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARIES)

if (CUDNN_FOUND)
  message(STATUS "Found CUDNN    (include: ${CUDNN_INCLUDE_DIR}, library: ${CUDNN_LIBRARIES})")
  mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARIES)
endif ()

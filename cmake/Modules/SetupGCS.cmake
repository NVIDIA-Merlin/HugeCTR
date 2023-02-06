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

function(gcs_setup)
  set(GC_CORE_VAR /usr/local/lib/libgoogle_cloud_cpp_common.so)
  set(GC_INTERNAL_VAR /usr/local/lib/libgoogle_cloud_cpp_rest_internal.so)
  set(GC_STORAGE_VAR /usr/local/lib/libgoogle_cloud_cpp_storage.so)
  get_filename_component(GC_CORE_VAR ${GC_CORE_VAR} ABSOLUTE)
  get_filename_component(GC_INTERNAL_VAR ${GC_INTERNAL_VAR} ABSOLUTE)
  get_filename_component(GC_STORAGE_VAR ${GC_STORAGE_VAR} ABSOLUTE)

  if(EXISTS "${GC_CORE_VAR}" AND EXISTS "${GC_INTERNAL_VAR}" AND EXISTS "${GC_STORAGE_VAR}")
    message(STATUS "GCS SDK is already installed, skip the installation!")
  else()
    message(STATUS "GCS SDK was not installed, will install it first!")
    execute_process(
      COMMAND /bin/bash ${PROJECT_SOURCE_DIR}/sbin/install-google-cloud-api.sh
      COMMAND_ECHO STDOUT
    )
  endif()

endfunction()
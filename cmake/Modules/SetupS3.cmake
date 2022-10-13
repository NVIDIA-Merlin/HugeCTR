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

function(s3_setup)
  set(AWS_CORE_VAR /usr/local/lib/libaws-cpp-sdk-core.so)
  set(AWS_S3_VAR /usr/local/lib/libaws-cpp-sdk-s3.so)
  get_filename_component(AWS_CORE_VAR ${AWS_CORE_VAR} ABSOLUTE)
  get_filename_component(AWS_S3_VAR ${AWS_S3_VAR} ABSOLUTE)

  if(EXISTS "${AWS_CORE_VAR}" AND EXISTS "${AWS_S3_VAR}")
    message(STATUS "AWS S3 SDK is already installed, skip the installation!")
  else()
    message(STATUS "AWS S3 SDK was not installed, will install it first!")
    execute_process(
      COMMAND /bin/bash ${PROJECT_SOURCE_DIR}/sbin/install-aws-sdk.sh
      COMMAND_ECHO STDOUT
    )
  endif()

endfunction()

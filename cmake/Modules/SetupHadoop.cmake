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

function(hadoop_setup hadoop_ver)
  set(HDFS_LIB_VAR /usr/local/lib/libhdfs.so)
  get_filename_component(HDFS_LIB_VAR ${HDFS_LIB_VAR} ABSOLUTE)
  if(EXISTS "${HDFS_LIB_VAR}")
    message(STATUS "HDFS is already installed, skip the installation of HDFS!")
  else()
    message(STATUS "HDFS was not installed, will install it first!")
    execute_process(
      COMMAND /bin/bash ${PROJECT_SOURCE_DIR}/sbin/build-hadoop.sh ${hadoop_ver}
      COMMAND_ECHO STDOUT
    )

    execute_process(
      COMMAND /bin/bash ${PROJECT_SOURCE_DIR}/sbin/install-hadoop.sh ${hadoop_ver}
      COMMAND_ECHO STDOUT
    )

  endif()

endfunction()

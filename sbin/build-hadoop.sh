#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
cd ${SCRIPT_DIR}/../third_party
HDFS_BUILD_MODE="$1"

# Build and install protobuf.
cd protobuf
./autogen.sh
./configure
make -j$(nproc)
make install
cd ..
ldconfig
echo "Protocol Buffers version: $(protoc --version)"

# Build and install Hadoop.
cd hadoop

# Temporarily disable name resolution for jboss repository. NVIDIA IT ticket number: "INC0866408"
if [[ ! "$(curl --connect-timeout 5 https://repository.jboss.org)" ]] 2>/dev/null; then
    echo 'Unable to connect to repository.jboss.org. Disabling...'
    echo '127.0.0.1 repository.jboss.org' >> /etc/hosts
fi

# Shorten compile, if only need client (e.g., CIs that do not need Hadoop).
if [[ "${HDFS_BUILD_MODE}" == "MINIMAL" ]]; then
  cd hadoop-hdfs-project/hadoop-hdfs-native-client
fi

# Build Hadoop.
mvn clean package \
  -Pdist,native \
  -DskipTests \
  -Dtar \
  -Dmaven.javadoc.skip=true \
  -Drequire.snappy \
  -Drequire.zstd \
  -Drequire.openssl \
  -Drequire.pmdk

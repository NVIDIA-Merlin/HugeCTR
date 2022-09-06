#!/usr/bin/env bash

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
cd ${SCRIPT_DIR}/../third_party

# Download, build and install protobuf.
cd protobuf
./autogen.sh
./configure
make -j$(nproc)
make install
cd ..
ldconfig
echo "Protocol Buffers version: $(protoc --version)"

# Download build and install Hadoop.
cd hadoop

# Temporarily disable name resolution for jboss repository. NVIDIA IT ticket number: "INC0866408"
if [[ ! "$(curl --connect-timeout 5 https://repository.jboss.org)" ]] 2>/dev/null; then
    echo 'Unable to connect to repository.jboss.org. Disabling...'
    echo '127.0.0.1 repository.jboss.org' >> /etc/hosts
fi

# Build Hadoop.
mvn clean package \
  -Pdist,native \
  -DskipTests \
  -Dtar -Dmaven.javadoc.skip=true \
  -Drequire.snappy \
  -Drequire.zstd \
  -Drequire.openssl \
  -Drequire.pmdk

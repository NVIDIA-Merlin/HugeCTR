#!/usr/bin/env bash

if [[ "$#" != 1 ]]; then
  echo "ERROR: Must provide Hadoop version number!"
  echo "       Example: ${BASH_SOURCE[0]} \"3.3.2\""
  exit 1
fi
HADOOP_VER="$1"

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
cd ${SCRIPT_DIR}

# Download desired revision.
git clone --branch rel/release-${HADOOP_VER} --depth 1 https://github.com/apache/hadoop.git hadoop
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

# Move compiled distribution to $SCRIPT_DIR and delete temporary files.
mv hadoop-dist/target/hadoop-${HADOOP_VER}.tar.gz ..
cd ..
rm -rf hadoop /root/.m2

# Self-delete.
rm -rf ${BASH_SOURCE[0]}
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

HDFS_BUILD_MODE="$1"
if [[ -z "${HADOOP_HOME}" ]]; then
  HADOOP_HOME=/opt/hadoop
fi


HADOOP_TAR=/tmp/hadoop-3.4.0.tar.gz
if [[ ! -f ${HADOOP_TAR} ]]; then
  wget https://dlcdn.apache.org/hadoop/common/hadoop-3.4.0/hadoop-3.4.0.tar.gz -O ${HADOOP_TAR}
  if [[ $? -ne 0 ]]; then
    echo "Failed to download Hadoop."
    exit 1
  fi
fi

# Extract files and delete archive.
mkdir -p ${HADOOP_HOME}/logs
tar zxf ${HADOOP_TAR} --strip-components 1 --directory ${HADOOP_HOME}

# Install header files if not yet installed.
mkdir -p ${HADOOP_HOME}/include
if [[ ! -f "${HADOOP_HOME}/include/hdfs.h" ]]; then
  cp hadoop-hdfs-project/hadoop-hdfs-native-client/src/main/native/libhdfs/include/hdfs/hdfs.h ${HADOOP_HOME}/include
fi

# Cleanup redundant files.
for f in $(find ${HADOOP_HOME} -name "*.cmd"); do
  rm -rf $f
done

# Pretend that the package has been installed like any other.
ln -sf ${HADOOP_HOME}/include/hdfs.h /usr/local/include/hdfs.h
ln -sf ${HADOOP_HOME}/lib/native/libhdfs.so /usr/local/lib/libhdfs.so
ln -sf ${HADOOP_HOME}/lib/native/libhdfs.so.0.0.0 /usr/local/lib/libhdfs.so.0.0.0
ln -sf ${HADOOP_HOME}/lib/native/libhadoop.so /usr/local/lib/libhadoop.so
ln -sf ${HADOOP_HOME}/lib/native/libhadoop.so.1.0.0 /usr/local/lib/libhadoop.so.1.0.0

# Create minimalist single-node "default" configuration.
# Skip if install with minimal mode.
if [[ "${HDFS_BUILD_MODE}" != "MINIMAL" ]]; then

  sed -i "s/^# export JAVA_HOME=$/export JAVA_HOME=${JAVA_HOME//\//\\\/}/g" ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

  echo '<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<!-- Single-node dummy configuration -->
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>' > ${HADOOP_HOME}/etc/hadoop/core-site.xml

  echo '<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<!-- Single-node dummy configuration -->
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
</configuration>' > ${HADOOP_HOME}/etc/hadoop/hdfs-site.xml

  if [[ -z $HOME/.ssh/id_ecdsa ]]; then
    ssh-keygen -q -t ecdsa -b 521 -N "" <<< ""
    cat $HOME/.ssh/id_ecdsa.pub >> $HOME/.ssh/authorized_keys
  fi

  ldconfig
  echo "
  Hadoop version: $(${HADOOP_HOME}/bin/hadoop version)

  To run a single-node hadoop instance (for development only):

    hadoop namenode -format
    service ssh start
    start-dfs.sh

  "
fi
#!/usr/bin/env bash

if [[ "$#" != 1 ]]; then
  echo "ERROR: Must provide Hadoop version number!"
  echo "       Example: ${BASH_SOURCE[0]} \"3.3.2\""
  exit 1
fi
HADOOP_VER="$1"

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
cd ${SCRIPT_DIR}
source hadoop.sh

# Extract files and delete archive.
mkdir -p ${HADOOP_HOME}/logs
tar xf hadoop-${HADOOP_VER}.tar.gz --strip-components 1 --directory ${HADOOP_HOME}
rm -rf hadoop-${HADOOP_VER}.tar.gz

# Cleanup reundant files.
for f in $(find ${HADOOP_HOME} -name *.cmd); do
  rm -rf $f
done

# Pretend that the package has been installed like any other.
ln -s ${HADOOP_HOME}/include/hdfs.h /usr/local/include/hdfs.h
ln -s ${HADOOP_HOME}/lib/native/libhdfs.so /usr/local/lib/libhdfs.so
ln -s ${HADOOP_HOME}/lib/native/libhdfs.so.0.0.0 /usr/local/lib/libhdfs.so.0.0.0
ln -s ${HADOOP_HOME}/lib/native/libhadoop.so /usr/local/lib/libhadoop.so
ln -s ${HADOOP_HOME}/lib/native/libhadoop.so.1.0.0 /usr/local/lib/libhadoop.so.1.0.0

# Create minimalist single-node "default" configuration.
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

ssh-keygen -q -t ecdsa -b 521 -N "" <<< ""
cat $HOME/.ssh/id_ecdsa.pub >> $HOME/.ssh/authorized_keys

ldconfig
echo "
Hadoop version: $(hadoop version)

To run a single-node hadoop instance (for development only):

    hadoop namenode -format
    service ssh start
    start-dfs.sh

"


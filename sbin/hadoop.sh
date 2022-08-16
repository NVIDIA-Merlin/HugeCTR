export HADOOP_HOME=/opt/hadoop

export PATH=${PATH}:${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin

export HDFS_NAMENODE_USER=root

export HDFS_SECONDARYNAMENODE_USER=root

export HDFS_DATANODE_USER=root

export YARN_RESOURCEMANAGER_USER=root

export YARN_NODEMANAGER_USER=root

export LIBHDFS_OPTS='-Djdk.lang.processReaperUseDefaultStackSize=true'

export UCX_ERROR_SIGNALS=''

export CLASSPATH=${CLASSPATH}:${HADOOP_HOME}/etc/hadoop/*:${HADOOP_HOME}/share/hadoop/common/*:${HADOOP_HOME}/share/hadoop/common/lib/*:${HADOOP_HOME}/share/hadoop/hdfs/*:${HADOOP_HOME}/share/hadoop/hdfs/lib/*:${HADOOP_HOME}/share/hadoop/mapreduce/*:${HADOOP_HOME}/share/hadoop/yarn/*:${HADOOP_HOME}/share/hadoop/yarn/lib/*


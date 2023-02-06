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


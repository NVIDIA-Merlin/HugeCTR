#!/bin/bash

echo $(pwd)
set -x

docker login -u ${CI_PRIVATE_USER} -p "${CI_PRIVATE_KEY}" "${CI_REGISTRY}"

ID=$(docker run --rm --name=hadoop_namenode -u root -dt gitlab-master.nvidia.com:5005/dl/hugectr/hugectr:devel_hps_thirdparties sh -c "/etc/init.d/ssh start && /usr/local/hadoop/sbin/start-dfs.sh && top")
docker logs $ID

ID=$(docker run --net=container:hadoop_namenode -u root -d ${CONT} bash -cx 'export CLASSPATH=$(hadoop classpath --glob) && cd /workdir/build/bin && ./hdfs_backend_test')
docker logs -f $ID
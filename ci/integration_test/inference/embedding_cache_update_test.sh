#!/bin/bash

echo $(pwd)
set -x

docker login -u ${CI_PRIVATE_USER} -p "${CI_PRIVATE_KEY}" "${CI_REGISTRY}"

ID=$(docker run --rm --net=host -u root -d gitlab-master.nvidia.com:5005/dl/hugectr/hugectr:devel_hps_thirdparties sh -c "cd /usr/local/redis/7000 && ../src/redis-server redis.conf ")
docker logs $ID

ID=$(docker run --rm --net=host -u root -d gitlab-master.nvidia.com:5005/dl/hugectr/hugectr:devel_hps_thirdparties sh -c "cd /usr/local/redis/7001 && ../src/redis-server redis.conf ")
docker logs $ID

ID=$(docker run --rm --net=host -u root -d gitlab-master.nvidia.com:5005/dl/hugectr/hugectr:devel_hps_thirdparties sh -c "cd /usr/local/redis/7002 && ../src/redis-server redis.conf ")
docker logs $ID

ID=$(docker run --rm --net=host -u root -d gitlab-master.nvidia.com:5005/dl/hugectr/hugectr:devel_hps_thirdparties sh -c "cd /usr/local/redis/src && echo yes | ./redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 --cluster-replicas 0 ")
docker logs $ID

ID=$(docker run --rm --net=host -u root -d gitlab-master.nvidia.com:5005/dl/hugectr/hugectr:devel_hps_thirdparties sh -c "export JAVA_HOME=/usr/local/jdk-16.0.2 && /usr/local/zookeeper/bin/zkServer.sh start && /usr/local/kafka/bin/kafka-server-start.sh /usr/local/kafka/config/server.properties ")
docker logs $ID

ID=$(docker run --net=host -v /mnt/nvdl/usr/aleliu/inference_ci/model_repository:/models -u root -d ${CONT} bash -cx "cd /workdir/build/bin && mkdir -p /hugectr/Test_Data/rockdb && ./embedding_cache_update_test || exit 1")
docker logs -f $ID
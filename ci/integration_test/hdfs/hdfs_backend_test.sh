#!/bin/bash

echo $(pwd)
set -ex

docker login -u ${CI_PRIVATE_USER} -p "${CI_PRIVATE_KEY}" "${CI_REGISTRY}"
ID=$(docker run --gpus all -d -u root ${CONT} bash -cx "\
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa && \
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
/etc/init.d/ssh start && \
hdfs namenode -format && \
bash /opt/hadoop/sbin/start-dfs.sh && \
cd /workdir/build/bin && \
./hdfs_backend_test && \
./file_loader_test")

docker logs -f $ID
exitCode=$(docker wait $ID)
docker rm $ID
exit $exitCode

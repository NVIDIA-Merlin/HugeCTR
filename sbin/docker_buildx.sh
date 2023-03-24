#!/bin/bash

########################################################################
# docker_buildx::docker_buildx() - Setup buildkit node, runs docker buildx, then tears down.
#                                  pass in all args as BUILDX_ARGS
#
# inputs:
#   CI_RUNNER_ID
#   BUILDX_ARGS
#
# eg: 
# CI_RUNNER_ID: 22310
# BUILDX ARGS:  --push --cache-from type=registry,ref=gitlab-master.nvidia.com:5005/dl/dgx/mxnet 
#               --cache-to type=registry,ref=gitlab-master.nvidia.com:5005/dl/dgx/mxnet,mode=max
#               -t gitlab-master.nvidia.com:5005/dl/dgx/mxnet:rluo-dockerbuild-refactor-py3.3827039-base-amd64 
#               -t gitlab-master.nvidia.com:5005/dl/dgx/mxnet:3827039-base-amd64 
#               --build-arg FROM_IMAGE_NAME=gitlab-master.nvidia.com:5005/dl/dgx/cuda:11.6-devel-ubuntu20.04--3708095
#               --build-arg FROM_SCRIPTS_IMAGE=gitlab-master.nvidia.com:5005/dl/devops/build-scripts:latest
#               --build-arg PYVER=3.8 --build-arg NVIDIA_BUILD_REF=611c6458bafc6c08e0dc6efd2a6e184d3f2fbd04
#               --build-arg NVIDIA_BUILD_ID=31430297 -f Dockerfile.base 
#               --build-arg NVIDIA_MXNET_VERSION=rluo-dockerbuild-refactor
#               --build-arg MXNET_VERSION=
########################################################################

docker_buildx::docker_buildx() {

  local in_CI_RUNNER_ID="$1"
  local in_BUILDX_ARGS="$2"

  echo "BUILDX ARGS: $in_BUILDX_ARGS"

  export DOCKER_CLI_EXPERIMENTAL=enabled

  cleanup()
  {
    docker stop "buildx_buildkit_node_${in_CI_RUNNER_ID}"
    docker rm "buildx_buildkit_node_${in_CI_RUNNER_ID}"
    docker buildx rm "buildkit_${in_CI_RUNNER_ID}"
  }

  trap_cleanup()
  {
    echo "Executing Trap"
    cleanup
    exit 1
  }

  trap trap_cleanup 1 2 3 6 9 15 19

  docker buildx create \
    --name "buildkit_${in_CI_RUNNER_ID}" \
    --node "node_${in_CI_RUNNER_ID}" \
    --config ./sbin/buildkitd.toml \
    --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=10485760 \
    --driver-opt env.BUILDKIT_STEP_LOG_MAX_SPEED=10485760 \
    --use

  docker buildx inspect --bootstrap

  docker buildx build $in_BUILDX_ARGS .
  RV=$?
  echo "exit code from the previous command -> $RV"
  cleanup
  return $RV 
}


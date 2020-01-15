#!/bin/bash

#cuda v10
#module load cuda/10.0
#module load cudnn/v7.6-cuda.10.0
#module load NCCL/2.4.8-1-cuda.10.0
#export CUDA=/public/apps/cuda/10.1
#export CUDNN=/public/apps/cudnn/v7.6/cuda
#export NCCL=/public/apps/NCCL/2.4.8-1

#cuda v9.2
module load cuda/9.2
module load cudnn/v7.3-cuda.9.2
module load NCCL/2.2.13-1-cuda.9.2
export CUDA=/public/apps/cuda/9.2
export CUDNN=/public/apps/cudnn/v7.3/cuda
export NCCL=/public/apps/NCCL/2.2.13-1

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCUDNN_INCLUDE_DIR=/public/apps/cudnn/v7.3/cuda/include -DCUDNN_LIBRARIES=/public/apps/cudnn/v7.3/cuda/lib64/libcudnn.so -DNCCL_INCLUDE_DIR=/public/apps/NCCL/2.4.8-1/include -DNCCL_LIBRARIES=/public/apps/NCCL/2.4.8-1/libnccl.so -DCUDNN_INC_PATHS=/public/apps/cudnn/v7.3/cuda/include -DSM=60 ..
make -j

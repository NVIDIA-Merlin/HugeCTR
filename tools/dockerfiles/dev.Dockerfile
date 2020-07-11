FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y \
    git \
    cmake \
    vim \
    wget \
    python3-pip

RUN python3 -m pip install --upgrade pip setuptools six && \
    python3 -m pip install --no-cache-dir \
    numpy \
    pandas \
    sklearn \
    ortools \
    tensorflow

RUN git clone https://github.com/NVIDIA/nccl.git --recursive && \
    cd nccl && \
    git checkout p2p && \
    make -j src.build

RUN mv nccl/build/include/*.h /usr/include && \
    mv nccl/build/lib/libnccl* /usr/lib/x86_64-linux-gnu/ && \
    rm -rf nccl

RUN echo 'export PS1="\s \w\$ "' >>/etc/bash.bashrc

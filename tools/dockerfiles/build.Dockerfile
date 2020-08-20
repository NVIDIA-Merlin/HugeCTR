FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
ARG CMAKE_BUILD_TYPE=Release
ARG SM="70;75;80"
ARG VAL_MODE=OFF
ARG ENABLE_MULTINODES=OFF
ARG NCCL_A2A=ON

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        lsb-release \
        libboost-all-dev \
        vim \
        wget \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# CMake version 3.14.3
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://cmake.org/files/v3.14/cmake-3.14.3-Linux-x86_64.sh && \
    mkdir -p /usr/local && \
    /bin/sh /var/tmp/cmake-3.14.3-Linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm -rf /var/tmp/cmake-3.14.3-Linux-x86_64.sh
ENV PATH=/usr/local/bin:$PATH

# pip
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        python3-wheel && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install numpy pandas sklearn ortools tensorflow 

RUN mkdir -p /opt/conda
ENV CONDA_PREFIX=/opt/conda

RUN if [ $(lsb_release --codename --short) = "stretch" ]; then \
      echo "deb http://deb.debian.org/debian $(lsb_release --codename --short)-backports main" >> /etc/apt/sources.list.d/backports.list; \ 
    fi && \
    wget https://apache.bintray.com/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb && \
    apt install -y -V ./apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb && \
    apt update && apt install -y libarrow-dev=0.17.1-1 libarrow-cuda-dev=0.17.1-1

# https://github.com/rapidsai/rmm.git
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch branch-0.15 https://github.com/rapidsai/rmm.git rmm && cd - && \
    cd /var/tmp/rmm && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/rmm

# CUDF 
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch branch-0.15 https://github.com/rapidsai/cudf.git cudf && cd - && \
    git clone --depth=1 --branch master https://github.com/dmlc/dlpack.git /var/tmp/dlpack && \
    cd /var/tmp/cudf/cpp && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGPU_ARCHS=$SM \
             -DARROW_INCLUDE_DIR=/usr/include/arrow/ -DDLPACK_INCLUDE=/var/tmp/dlpack/include \
             -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDISABLE_DEPRECATION_WARNING=ON && \ 
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/dlpack /var/tmp/cudf

RUN git clone https://github.com/NVIDIA/HugeCTR.git HugeCTR &&\
    cd HugeCTR && \
    git submodule update --init --recursive && \
    mkdir build && cd build &&\
    cmake -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DSM=$SM \
          -DVAL_MODE=$VAL_MODE -DENABLE_MULTINODES=$ENABLE_MULTINODES -DNCCL_A2A=NCCL_A2A .. && \
    make -j$(nproc) &&\
    mkdir /usr/local/hugectr &&\
    make install &&\
    chmod +x /usr/local/hugectr/bin/* &&\
    rm -rf HugeCTR 
ENV PATH /usr/local/hugectr/bin:$PATH



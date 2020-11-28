FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04 AS devel

ARG SM="60;61;70;75;80"

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        vim \
        wget \
        make \
        software-properties-common \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        lsb-release \
        libboost-all-dev \
        zlib1g-dev && \
    add-apt-repository ppa:git-core/ppa -y && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# CMake-3.17.0
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.sh && \
    mkdir -p /usr/local && \
    /bin/sh /var/tmp/cmake-3.17.0-Linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm -rf /var/tmp/cmake-3.17.0-Linux-x86_64.sh
ENV PATH=/usr/local/bin:$PATH

# pip
RUN echo alias python='/usr/bin/python3' >> /etc/bash.bashrc && \
    pip3 install --upgrade pip && \
    pip3 install numpy pandas sklearn ortools jupyter tf-nightly-gpu

# UCX-1.8.0
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils-dev \
        file \
        libnuma-dev && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/openucx/ucx/releases/download/v1.8.0/ucx-1.8.0.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/ucx-1.8.0.tar.gz -C /var/tmp -z && \
    cd /var/tmp/ucx-1.8.0 &&   ./configure --prefix=/usr/local/ucx --disable-assertions --disable-debug --disable-doxygen-doc --disable-logging --disable-params-check --enable-optimizations --with-cuda=/usr/local/cuda && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/ucx-1.8.0 /var/tmp/ucx-1.8.0.tar.gz
ENV CPATH=/usr/local/ucx/include:$CPATH \
    LD_LIBRARY_PATH=/usr/local/ucx/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/ucx/lib:$LIBRARY_PATH \
    PATH=/usr/local/ucx/bin:$PATH

# HWLOC-2.2.0
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://download.open-mpi.org/release/hwloc/v2.2/hwloc-2.2.0.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/hwloc-2.2.0.tar.gz -C /var/tmp -z && \
    cd /var/tmp/hwloc-2.2.0 && ./configure --prefix=/usr/local/hwloc && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/hwloc-2.2.0 /var/tmp/hwloc-2.2.0.tar.gz
ENV CPATH=/usr/local/hwloc/include:$CPATH \
    LD_LIBRARY_PATH=/usr/local/hwloc/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/hwloc/lib:$LIBRARY_PATH \
    PATH=/usr/local/hwloc/bin:$PATH

# OpenMPI-4.0.3
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bzip2 \
        file \
        libnuma-dev \
        openssh-client \
        perl \
        tar && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.3.tar.bz2 && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/openmpi-4.0.3.tar.bz2 -C /var/tmp -j && \
    cd /var/tmp/openmpi-4.0.3 &&   ./configure --prefix=/usr/local/openmpi --disable-getpwuid --enable-orterun-prefix-by-default --with-cuda --with-ucx=/usr/local/ucx --with-verbs && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/openmpi-4.0.3 /var/tmp/openmpi-4.0.3.tar.bz2
ENV LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/openmpi/bin:$PATH

# MPI for python
RUN env MPICC=/usr/local/openmpi/bin pip install mpi4py

RUN mkdir -p /opt/conda
ENV CONDA_PREFIX=/opt/conda \
    NCCL_LAUNCH_MODE=PARALLEL

RUN if [ $(lsb_release --codename --short) = "stretch" ]; then \
      echo "deb http://deb.debian.org/debian $(lsb_release --codename --short)-backports main" >> /etc/apt/sources.list.d/backports.list; \
    fi && \
    wget https://apache.bintray.com/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb && \
    apt install -y -V ./apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb && \
    apt update && apt install -y libarrow-dev libarrow-cuda-dev && \
    dpkg -r --force-depends libnvidia-compute-450-server

# RMM-0.16
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch branch-0.16 https://github.com/rapidsai/rmm.git rmm && cd - && \
    cd /var/tmp/rmm && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/rmm

# CUDF-0.16
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch branch-0.16 https://github.com/rapidsai/cudf.git cudf && cd - && \
    git clone --depth=1 --branch main https://github.com/dmlc/dlpack.git /var/tmp/dlpack && \
    cd /var/tmp/cudf/cpp && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGPU_ARCHS=$SM \
             -DARROW_INCLUDE_DIR=/usr/include/arrow/ -DDLPACK_INCLUDE=/var/tmp/dlpack/include \
             -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDISABLE_DEPRECATION_WARNING=ON && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/dlpack /var/tmp/cudf

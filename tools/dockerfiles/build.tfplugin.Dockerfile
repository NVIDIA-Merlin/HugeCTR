FROM nvcr.io/nvidia/tensorflow:20.09-tf2-py3

ARG SM="60;61;70;75;80"

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get remove cmake git -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
        lsb-release \
        libboost-all-dev \
        zlib1g-dev \
        openssh-client && \
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
ENV PATH=/root/.local/bin:$PATH
RUN python3 -m pip install --user --upgrade pip && \
    pip3 install --user pandas sklearn ortools scikit-learn

# Clear MPI and UCX in base image
RUN PATH=$(REMOVE_PART="/usr/local/mpi/bin" sh -c 'echo ":$PATH:" | sed "s@:$REMOVE_PART:@:@g;s@^:\(.*\):\$@\1@"') && \
    PATH=$(REMOVE_PART="/usr/local/ucx/bin" sh -c 'echo ":$PATH:" | sed "s@:$REMOVE_PART:@:@g;s@^:\(.*\):\$@\1@"') && \
    rm -rf /usr/local/mpi /usr/local/ucx

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
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.3.tar.bz2 && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/openmpi-4.0.3.tar.bz2 -C /var/tmp -j && \
    cd /var/tmp/openmpi-4.0.3 &&   ./configure --prefix=/usr/local/openmpi --disable-getpwuid --enable-orterun-prefix-by-default --with-cuda --with-ucx=/usr/local/ucx && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/openmpi-4.0.3 /var/tmp/openmpi-4.0.3.tar.bz2
ENV LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/openmpi/bin:$PATH

# MPI for python
RUN env MPICC=/usr/local/openmpi/bin pip install mpi4py

# NCCL
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/NVIDIA/nccl/archive/v2.7.6-1.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/v2.7.6-1.tar.gz -C /var/tmp -z && \
    cd /var/tmp/nccl-2.7.6-1 && \
    PREFIX=/usr/local/nccl make -j$(nproc) install && \
    rm -rf /var/tmp/nccl-2.7.6-1 /var/tmp/v2.7.6-1.tar.gz
ENV CPATH=/usr/local/nccl/include:$CPATH \
    LD_LIBRARY_PATH=/usr/local/nccl/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/nccl/lib:$LIBRARY_PATH \
    PATH=/usr/local/nccl/bin:$PATH

RUN mkdir -p /opt/conda
ENV CONDA_PREFIX=/opt/conda \
    NCCL_LAUNCH_MODE=PARALLEL

RUN if [ $(lsb_release --codename --short) = "stretch" ]; then \
      echo "deb http://deb.debian.org/debian $(lsb_release --codename --short)-backports main" >> /etc/apt/sources.list.d/backports.list; \
    fi && \
    wget https://apache.bintray.com/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb && \
    apt install -y -V ./apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb && \
    apt update && apt install -y libarrow-dev=0.17.1-1 libarrow-cuda-dev=0.17.1-1

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

# HugeCTR
RUN git clone https://github.com/NVIDIA/HugeCTR.git HugeCTR &&\
    cd HugeCTR && \
    git submodule update --init --recursive && \
    mkdir build && cd build &&\
    cmake -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DSM=$SM \
          -DVAL_MODE=$VAL_MODE -DENABLE_MULTINODES=$ENABLE_MULTINODES -DNCCL_A2A=$NCCL_A2A .. && \
    make -j$(nproc) &&\
    mkdir /usr/local/hugectr &&\
    make install &&\
    chmod +x /usr/local/hugectr/bin/* &&\
    chmod +x /usr/local/hugectr/lib/* &&\
    rm -rf HugeCTR
ENV PATH /usr/local/hugectr/bin:$PATH
ENV PYTHONPATH /usr/local/hugectr/lib:$PYTHONPATH

# ENTRYPOINT [ "echo", "\nWelcome to HugeCTR docker!\n" ]

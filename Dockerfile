FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ARG SM_VERSION

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y \
    git \
    cmake \
    autoconf \
    libtool \
    libnuma-dev \
    flex

WORKDIR /
RUN git clone https://github.com/openucx/ucx.git && \
    cd ucx && \
    ./autogen.sh && \
    ./contrib/configure-release && \
    make -j install && \
    make clean

WORKDIR /
RUN git clone https://github.com/open-mpi/hwloc.git && \
    cd hwloc && \
    ./autogen.sh && \
    ./configure && \
    make -j install && \
    make clean

WORKDIR /
RUN git clone --recursive https://github.com/open-mpi/ompi.git && \
    cd ompi && \
    ./autogen.pl && \
    ./configure --with-ucx=/usr --with-hwloc=/usr/local && \
    make -j install && \
    make clean

WORKDIR /
RUN git clone https://gitlab-master.nvidia.com/zehuanw/hugectr.git && \
    cd hugectr && \
    git submodule update --init --recursive

WORKDIR /hugectr
RUN mkdir -p build && \
    cd build && \
    cmake -DENABLE_MULTINODES=ON -DCMAKE_BUILD_TYPE=Release -DSM=${SM_VERSION} .. && \
    make -j

ENV PATH="/hugectr/build/bin:${PATH}"
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

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
RUN rm -rf ucx && \
    rm -rf hwloc && \
    rm -rf ompi

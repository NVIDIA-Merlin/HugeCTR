FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y \
    git \
    cmake \
    vim \
    wget \
    clang-format \
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

RUN apt-get install -y curl
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya

ENV PATH $PATH:/opt/conda/bin
RUN conda install -c rapidsai-nightly -c nvidia -c conda-forge -c defaults cudf=0.15 python=3.7 cudatoolkit=10.2
#RUN conda install -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.14 python=3.6
RUN rm /opt/conda/include/nccl.h && rm /opt/conda/lib/libnccl.so
ENV CONDA_PREFIX=/opt/conda
RUN echo 'export PS1="\s \w\$ "' >>/etc/bash.bashrc

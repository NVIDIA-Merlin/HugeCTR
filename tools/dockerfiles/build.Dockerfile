FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04-rc
ARG CMAKE_BUILD_TYPE=Release
ARG SM="70;80"
ARG VAL_MODE=OFF
ARG ENABLE_MULTINODES=OFF
ARG NCCL_A2A=ON

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y \
    git \
    cmake \
    vim \
    wget \
    python3-pip

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    numpy \
    pandas \
    sklearn \
    ortools \
    tensorflow

RUN cp /usr/local/cuda/lib64/libnccl*  /usr/lib/x86_64-linux-gnu/ && \
    cp /usr/local/cuda-11.0/targets/x86_64-linux/include/nccl*.h  /usr/include

RUN echo 'export PS1="\s \w\$ "' >>/etc/bash.bashrc

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
    
RUN git clone https://github.com/NVIDIA/HugeCTR.git &&\
    cd HugeCTR && \
    git submodule update --init --recursive && \
    mkdir build && cd build &&\
    cmake -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DSM=$SM \
    -DVAL_MODE=$VAL_MODE -DENABLE_MULTINODES=$ENABLE_MULTINODES -DNCCL_A2A=NCCL_A2A .. && \
    make -j &&\
    mkdir /usr/local/hugectr &&\
    make install &&\
    chmod +x /usr/local/hugectr/bin/* &&\
    rm -rf HugeCTR 

ENV PATH /usr/local/hugectr/bin:$PATH



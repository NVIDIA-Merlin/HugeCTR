FROM nvcr.io/nvidia/tritonserver:21.04-py3 AS devel

ARG SM="70;75;80"
ARG RELEASE=false

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git gdb make wget curl \
        python3-pip python3-setuptools python3-wheel \
        zlib1g-dev lsb-release rapidjson-dev ca-certificates libboost-all-dev software-properties-common && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install pandas sklearn && \
    pip3 cache purge

# CMake-3.19.6
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://cmake.org/files/v3.19/cmake-3.19.6-Linux-x86_64.sh && \
    mkdir -p /usr/local && \
    /bin/sh /var/tmp/cmake-3.19.6-Linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm -rf /var/tmp/cmake-3.19.6-Linux-x86_64.sh
ENV PATH=/usr/local/bin:$PATH

# NCCL-p2p
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch p2p https://github.com/NVIDIA/nccl.git nccl && cd - && \
    cd /var/tmp/nccl && \
    PREFIX=/usr/local/nccl make -j$(nproc) install && \
    rm -rf /var/tmp/nccl
ENV CPATH=/usr/local/nccl/include:$CPATH \
    LD_LIBRARY_PATH=/usr/local/nccl/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/nccl/lib:$LIBRARY_PATH \
    PATH=/usr/local/nccl/bin:$PATH

RUN mkdir -p /opt/conda
ENV CONDA_PREFIX=/opt/conda

# RMM-0.19
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch branch-0.19 https://github.com/rapidsai/rmm.git rmm && cd - && \
    cd /var/tmp/rmm && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DBUILD_TESTS=OFF && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/rmm

# HugeCTR Inference
RUN if [ "$RELEASE" = "true" ]; \
    then \
      mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch master https://github.com/NVIDIA/HugeCTR.git HugeCTR && cd - && \
      cd /var/tmp/HugeCTR && \
      git submodule update --init --recursive && \
      mkdir -p build && cd build &&\
      cmake -DCMAKE_BUILD_TYPE=Release -DSM=$SM -DENABLE_INFERENCE=ON .. && make -j$(nproc) && make install && \
      export CPATH=/usr/local/hugectr/include:$CPATH && \
      export LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH && \
      cd /var/tmp && git clone --depth=1 --branch main https://github.com/triton-inference-server/hugectr_backend hugectr_inference_backend && cd - && \
      cd /var/tmp/hugectr_inference_backend && \
      mkdir -p build && cd build && \
      cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/hugectr .. && make -j$(nproc) && make install && \
      rm -rf /var/tmp/HugeCTR /var/tmp/hugectr_inference_backend; \
    else \
      echo "Build container for development successfully"; \
    fi
ENV CPATH=/usr/local/hugectr/include:$CPATH \ 
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/hugectr/bin:$PATH

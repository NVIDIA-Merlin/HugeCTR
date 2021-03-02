FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 AS devel

ARG RELEASE=false
ARG SM="60;61;70;75;80"

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        vim gdb git wget tar curl python-dev python3-dev \
        zlib1g-dev lsb-release ca-certificates clang-format libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp http://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh && \
    bash /var/tmp/Miniconda3-4.7.12-Linux-x86_64.sh -b -p /opt/conda && \
    /opt/conda/bin/conda init && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda update -n base -c defaults conda && \
    conda config --add channels conda-forge --add channels nvidia --add channels rapidsai --add channels anaconda && \
    conda install -y cmake=3.19.6 pip cudnn=8.0.4 rmm=0.18 cudatoolkit=11.0 && \
    /opt/conda/bin/conda clean -afy && \
    rm -rf /var/tmp/Miniconda3-4.7.12-Linux-x86_64.sh && \
    rm -rf /opt/conda/include/nccl.h /opt/conda/lib/libnccl.so /opt/conda/include/google
ENV CPATH=/opt/conda/include:$CPATH \
    LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/opt/conda/lib:$LIBRARY_PATH \
    PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda \
    NCCL_LAUNCH_MODE=PARALLEL

RUN pip3 install numpy==1.19.2 pandas sklearn ortools nvtx-plugins jupyter tensorflow==2.4.0 && \
    pip3 cache purge

RUN if [ "$RELEASE" = "true" ]; \
    then \
      mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch master https://github.com/NVIDIA/HugeCTR.git HugeCTR && cd - && \
      cd /var/tmp/HugeCTR && \
      git submodule update --init --recursive && \
      mkdir -p build && cd build && \
      cmake -DCMAKE_BUILD_TYPE=Release -DSM=$SM -DONLY_EMB_PLUGIN=ON .. && make -j$(nproc) && make install && \
      rm -rf /var/tmp/HugeCTR; \
    else \
      echo "Build container for development successfully"; \
    fi
ENV LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH

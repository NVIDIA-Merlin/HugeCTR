FROM nvcr.io/nvidia/merlin/merlin-tensorflow:22.07

ARG HIREDIS_VER=1.0.2
ARG REDIS_PP_VER=1.3.3
ARG ROCKSDB_VER=6.29.3
ARG RDKAFKA_VER=1.8.2

# Dependency: Build and install Redis native client.
RUN git clone --branch v${HIREDIS_VER} --depth 1 https://github.com/redis/hiredis.git hiredis && \
    mkdir hiredis/build && \
    cd hiredis/build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf hiredis

RUN git clone --branch ${REDIS_PP_VER} --depth 1 https://github.com/sewenew/redis-plus-plus.git redis_pp && \
    mkdir redis_pp/build && \
    cd redis_pp/build && \
    cmake -DREDIS_PLUS_PLUS_CXX_STANDARD=17 .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf redis_pp

# Dependency: Build and install RocksDB.
RUN git clone --branch v${ROCKSDB_VER} --depth 1 https://github.com/facebook/rocksdb.git rocksdb && \
    cd rocksdb && \
    PORTABLE=1 make -j$(nproc) shared_lib && \
    make install-shared && \
    cd .. && \
    rm -rf rocksdb

# Dependency: Build and install RdKafka.
RUN git clone --branch v"${RDKAFKA_VER}" --depth 1 https://github.com/edenhill/librdkafka.git rdkafka && \
    cd rdkafka && \
    ./configure --enable-static && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf rdkafka

RUN apt update -y --fix-missing && \
    apt install -y --no-install-recommends \
       libxxhash-dev

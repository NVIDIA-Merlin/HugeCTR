#!/bin/bash
hugectr_base=$(pwd)
cd third_party/rmm/ && git pull origin branch-0.15 && git checkout origin/branch-0.15 && mkdir -p build && cd build &&\
 cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX && make -j && make install && cd ../../..
pwd
if [[ -z "${GPU_ARCH}" ]]; then
  CUDF_GPU_ARCH="70"
else
 CUDF_GPU_ARCH="${GPU_ARCH}"
fi

cd third_party/cudf/cpp && git pull origin branch-0.15 && git checkout origin/branch-0.15 && mkdir -p build && cd build &&\
 cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGPU_ARCHS=$CUDF_GPU_ARCH -DARROW_INCLUDE_DIR=/usr/include/arrow/ -DDLPACK_INCLUDE=$hugectr_base/third_party/dlpack/include -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDISABLE_DEPRECATION_WARNING=ON &&\
 make -j && make install && cd ../../../..

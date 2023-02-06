#!/usr/bin/env bash

#Install absl
git clone https://github.com/abseil/abseil-cpp.git

cd abseil-cpp && mkdir -p build && cd build
cmake -DBUILD_SHARED_LIBS=ON .. && make install
cd ../..

#Install crc32c
git clone https://github.com/google/crc32c.git

cd crc32c && git submodule update --init --recursive
mkdir -p build && cd build
cmake -DCRC32C_BUILD_TESTS=0 -DCRC32C_BUILD_BENCHMARKS=0 .. && make all install
cd ../..

#Clone the google-cloud-cpp git repo
git clone https://github.com/googleapis/google-cloud-cpp.git

cd google-cloud-cpp
mkdir -p build
cd build
cmake -DCMAKE_CXX_FLAGS=-fPIC -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DGOOGLE_CLOUD_CPP_ENABLE_EXAMPLES=OFF -DGOOGLE_CLOUD_CPP_ENABLE=storage ..
make -j install

cd ../..
rm -rfv abseil-cpp
rm -rfv crc32c
rm -rfv google-cloud-cpp
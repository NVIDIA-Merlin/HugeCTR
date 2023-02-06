#!/usr/bin/env bash

#Clone the aws-sdk-cpp git repo
git clone --branch 1.10.0 --depth 1 --recurse-submodules https://github.com/aws/aws-sdk-cpp.git aws-sdk-cpp

#Build aws-cpp-sdk- with only the s3 components
mkdir -p build
cd build
cmake ../aws-sdk-cpp -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY="s3" -DENABLE_TESTING=OFF
make -j install

#Remove the repo
cd ..
rm -rf build
rm -rf aws-sdk-cpp

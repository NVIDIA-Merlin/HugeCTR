#!/usr/bin/env bash

#Clone the aws-sdk-cpp git repo
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git aws-sdk-cpp

#Build aws-cpp-sdk- with only the s3 components
mkdir -p build
cd build
cmake ../aws-sdk-cpp -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY="s3"
make -j install

#Remove the repo
cd ..
rm -rfv build
rm -rfv aws-sdk-cpp
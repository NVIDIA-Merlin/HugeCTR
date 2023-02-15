#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#Install absl
git clone https://github.com/abseil/abseil-cpp.git

cd abseil-cpp && mkdir -p build && cd build
cmake -DBUILD_SHARED_LIBS=ON .. && make install
cd ../..

#Install crc32c
git clone --recurse-submodules https://github.com/google/crc32c.git

cd crc32c && mkdir -p build && cd build
cmake -DCRC32C_BUILD_TESTS=0 -DCRC32C_BUILD_BENCHMARKS=0 .. && make all install
cd ../..

#Clone the google-cloud-cpp git repo
git clone --branch v2.7.0 --depth 1 https://github.com/googleapis/google-cloud-cpp.git

cd google-cloud-cpp && mkdir -p build && cd build
cmake -DCMAKE_CXX_FLAGS=-fPIC -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DGOOGLE_CLOUD_CPP_ENABLE_EXAMPLES=OFF -DGOOGLE_CLOUD_CPP_ENABLE=storage ..
make -j install

cd ../..
rm -rf abseil-cpp
rm -rf crc32c
rm -rf google-cloud-cpp
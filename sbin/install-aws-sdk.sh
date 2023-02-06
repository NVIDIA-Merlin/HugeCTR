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

"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import sys

from setuptools import find_packages
from skbuild import setup


def get_cmake_args():
    gpu_capabilities = ["70", "75", "80", "90"]
    if os.getenv("HPS_COMPILE_GPU_SM"):
        gpu_capabilities = os.getenv("HPS_COMPILE_GPU_SM")
        gpu_capabilities = str(gpu_capabilities).strip().split(";")

    cmake_build_type = "Release"
    if os.getenv("HPS_COMPILE_BUILD_TYPE"):
        cmake_build_type = (
            "Debug"
            if os.getenv("HPS_COMPILE_BUILD_TYPE") in ["DEBUG", "debug", "Debug"]
            else "Release"
        )

    cmake_args = [
        "-DSM='{}'".format(";".join(gpu_capabilities)),
        "-DCMAKE_BUILD_TYPE={}".format(cmake_build_type),
        "-DPYTHON_VERSION=python3.10",
    ]
    return cmake_args


setup(
    name="merlin-hps",
    author="NVIDIA",
    author_email="hugectr-dev@exchange.nvidia.com",
    url="https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/hps_torch",
    license="Apache 2.0",
    platforms=["Linux"],
    python_requires=">=3",
    packages=find_packages(),
    cmake_args=get_cmake_args(),
)

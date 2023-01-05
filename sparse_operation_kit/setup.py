"""
 Copyright (c) 2021, NVIDIA CORPORATION.

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

REQUIRED_PACKAGES = [
    "horovod>=0.26.1",
    "scikit-build >= 0.16.3",
]


def _GetSOKVersion():
    _version_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sparse_operation_kit/core/"
    )
    sys.path.append(_version_path)
    from _version import __version__

    version = __version__
    del __version__
    sys.path.pop(-1)
    return version


def get_cmake_args():
    gpu_capabilities = ["70", "75", "80"]
    if os.getenv("SOK_COMPILE_GPU_SM"):
        gpu_capabilities = os.getenv("SOK_COMPILE_GPU_SM")
        gpu_capabilities = str(gpu_capabilities).strip().split(";")

    use_nvtx = "OFF"
    if os.getenv("SOK_COMPILE_USE_NVTX"):
        use_nvtx = "ON" if os.getenv("SOK_COMPILE_USE_NVTX") in ["1", "ON", "On", "on"] else "OFF"

    dedicated_cuda_stream = "ON"
    if os.getenv("SOK_COMPILE_ASYNC"):
        dedicated_cuda_stream = (
            "OFF" if os.getenv("SOK_COMPILE_ASYNC") in ["0", "OFF", "Off", "off"] else "ON"
        )

    unit_test = "OFF"
    if os.getenv("SOK_COMPILE_UNIT_TEST"):
        unit_test = "ON" if os.getenv("SOK_COMPILE_UNIT_TEST") in ["1", "ON", "On", "on"] else "OFF"

    cmake_build_type = "Release"
    if os.getenv("SOK_COMPILE_BUILD_TYPE"):
        cmake_build_type = (
            "Debug"
            if os.getenv("SOK_COMPILE_BUILD_TYPE") in ["DEBUG", "debug", "Debug"]
            else "Release"
        )

    enable_deeprec = "OFF"
    if os.getenv("ENABLE_DEEPREC"):
        enable_deeprec = (
            "OFF" if os.getenv("ENABLE_DEEPREC") in ["0", "OFF", "Off", "off"] else "ON"
        )
    cmake_args = [
        "-DSM='{}'".format(";".join(gpu_capabilities)),
        "-DUSE_NVTX={}".format(use_nvtx),
        "-DSOK_ASYNC={}".format(dedicated_cuda_stream),
        "-DSOK_UNIT_TEST={}".format(unit_test),
        "-DCMAKE_BUILD_TYPE={}".format(cmake_build_type),
        "-DENABLE_DEEPREC={}".format(enable_deeprec),
    ]
    return cmake_args


# We haven't found a proper way to maintain the directory structure of
# the parent folder(i.e. HugeCTR) when using skbuild to make pip package,
# so we use a workaround here: copy the content of parent folder into
# sparse_operation_kit/ before making pip package.
os.system("cp -r ../HugeCTR ./")
os.system("mkdir third_party")
os.system("cp -r ../third_party/json ./third_party/")


setup(
    name="merlin-sok",
    version=_GetSOKVersion(),
    author="NVIDIA",
    author_email="hugectr-dev@exchange.nvidia.com",
    url="https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/sparse_operation_kit",
    description="SparseOperationKit (SOK) is a python package wrapped GPU accelerated"
    " operations dedicated for sparse training / inference cases.",
    long_description="SparseOperationKit (SOK) is a python package wrapped GPU accelerated "
    "operations dedicated for sparse training / inference cases. "
    "It is designed to be compatible with common DeepLearning (DL) frameworks, "
    "for instance, TensorFlow. "
    "Most of the algorithm implementations in SOK are extracted from HugeCTR, "
    "which is a GPU-accelerated recommender framework designed to distribute "
    "training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs). "
    "If you are looking for a very efficient solution for CTRs, please check HugeCTR.",
    extras_require={"tensorflow": "tensorflow>=1.15"},
    license="Apache 2.0",
    platforms=["Linux"],
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3",  # TODO: make it compatible with python2.7
    packages=find_packages(),
    cmake_args=get_cmake_args(),
)

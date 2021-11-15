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

from setuptools import setup
from setuptools import find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import os, sys
import subprocess

def _GetSOKVersion():
    _version_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "sparse_operation_kit/core/")
    sys.path.append(_version_path)
    from _version import __version__
    version = __version__
    del __version__
    sys.path.pop(-1)
    return version

class SOKExtension(Extension):
    def __init__(self, name, cmake_file_path="./", sources=[], **kwargs):
        Extension.__init__(self, name, sources=sources, **kwargs)
        self._CMakeLists_dir = os.path.abspath(cmake_file_path)

class SOKBuildExtension(build_ext):
    def build_extensions(self):
        if os.getenv("SOK_NO_COMPILE") == "1":
            # skip compile the source codes
            return
        
        gpu_capabilities = ["70", "75", "80"]
        if os.getenv("SOK_COMPILE_SM"):
            gpu_capabilities = str(gpu_capabilities).strip().split(";")

        use_nvtx = "OFF"
        if os.getenv("SOK_COMPILE_USE_NVTX"):
            use_nvtx = "ON" if "1" == os.getenv("SOK_COMPILE_USE_NVTX") else "OFF"

        dedicated_cuda_stream = "ON"
        if os.getenv("SOK_COMPILE_ASYNC"):
            dedicated_cuda_stream = "ON" if os.getenv("SOK_COMPILE_USE_NVTX") in ["1", "ON", "On", "on"] else "OFF"

        cmake_args = ["-DSM='{}'".format(";".join(gpu_capabilities)),
                      "-DUSE_NVTX={}".format(use_nvtx),
                      "-DSOK_ASYNC={}".format(dedicated_cuda_stream)]
        cmake_args = " ".join(cmake_args)

        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        for extension in self.extensions:
            # ext_fullname = self.get_ext_fullname(extension.name)
            # ext_fullpath = self.get_ext_fullpath(extension.name)
            # print("[INFO]: ext_fullname = ", ext_fullname)
            # print("[INFO]: ext_fullpath = ", ext_fullpath)
            try:
                subprocess.check_call("cmake {} {} && make -j && make install".format(cmake_args, 
                                                                    extension._CMakeLists_dir), 
                                    shell=True,
                                    cwd=build_dir)
            except OSError as error:
                raise RuntimeError("Compile SOK faild, due to {}".format(str(error)))

        if sys.argv[1].startswith("develop"):
            # remove unfound so
            self.extensions = [ext for ext in self.extensions
                                if os.path.exists(self.get_ext_fullpath(ext.name))]

setup(
    name="SparseOperationKit",
    version=_GetSOKVersion(),
    author="NVIDIA",
    url="https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/sparse_operation_kit",
    extras_require={"tensorflow": "tensorflow>=1.15"}, # TODO: should use install_requires??
    python_requires='>=3', # TODO: make it compatible with python2.7
    packages=find_packages(
        where="./",
        include=["sparse_operation_kit"],
        exclude=[]
    ),
    package_dir={"": "./"},
    cmdclass={"sok_build_extension": SOKBuildExtension},
    ext_modules=[SOKExtension("sparse_operation_kit")],
)

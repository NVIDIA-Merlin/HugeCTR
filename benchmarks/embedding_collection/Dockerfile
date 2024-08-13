# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

FROM nvcr.io/nvidia/pytorch:23.09-py3

WORKDIR /workspace/dlrm

RUN cd /workspace/dlrm && \
    git clone --recursive -b v0.6.0-rc2 https://github.com/pytorch/FBGEMM.git && \
    cd FBGEMM/fbgemm_gpu && pip install -r requirements.txt && \
    python setup.py install -DTORCH_CUDA_ARCH_LIST="7.5;8.0;9.0" && rm -rf /workspace/dlrm/FBGEMM

RUN pip install torchmetrics==1.1.0 pyre_extensions iopath
RUN cd /workspace/dlrm && \
    git clone --recursive -b v0.6.0-rc2 https://github.com/pytorch/torchrec.git && \
    cd torchrec && \
    python3 setup.py install && rm -rf /workspace/dlrm/torchrec


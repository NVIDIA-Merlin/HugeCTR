HugeCTR Software Stack
----------------------

This dependency matrix shows the software stack of NVIDIA HugeCTR project. The matrix provides a single view into the supported software and specific versions for each release of the HugeCTR. Software stacks included are:

* <a href="#md_cap_train">train.Dockerfile</a>
* <a href="#md_cap_plugin">embedding-plugin.Dockerfile</a>

**Important**: Content that is included in <<>> brackets indicates new content from the previously published version.

<div align="center"><a name="md_cap_train">Table 1. Software stack for HugeCTR training image (train.Dockerfile)</a></div>

| Container Image                                              | v3.0                                                         | v2.3                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **DGX**                                                      |                                                              |                                                              |
| DGX System                                                   | DGX-1<br>DGX-2<br>DGX A100<br>DGX Station                    | DGX-1<br>DGX-2<br>DGX A100<br>DGX Station                    |
| **NVIDIA Certified Systems**                                 |                                                              |                                                              |
| [NVIDIA Driver](http://www.nvidia.com/Download/index.aspx?lang=en-us) | Release v3.0 is based on CUDA 11.0.221, which requires NVIDIA driver release 450.51.<br><br>However, if you are running on Tesla (for example, T4 or any other Tesla board), you may use NVIDIA driver release 418.xx or 440.30.<br><br>The CUDA driver's compatibility package only supports particular drivers. <a href="#md_idx_1">1</a> | Release v2.3 is based on CUDA 11.0.221, which requires NVIDIA driver release 450.51.<br><br>However, if you are running on Tesla (for example, T4 or any other Tesla board), you may use NVIDIA driver release 418.xx or 440.30.<br><br>The CUDA driver's compatibility package only supports particular drivers. <a href="#md_idx_1">1</a> |
| GPU Model                                                    | [NVIDIA Ampere GPU Architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture)<br>[Turing](https://www.nvidia.com/en-us/geforce/turing/)<br>[Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)<br>[Pascal](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) | [NVIDIA Ampere GPU Architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture)<br>[Turing](https://www.nvidia.com/en-us/geforce/turing/)<br>[Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)<br>[Pascal](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) |
| **Base Container Image**                                     |                                                              |                                                              |
| Container OS                                                 | [Ubuntu 18.04](http://releases.ubuntu.com/18.04/)            | [Ubuntu 18.04](http://releases.ubuntu.com/18.04/)            |
| [CUDA](http://docs.nvidia.com/cuda/index.html)               | [11.0.221](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) | [11.0.221](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) |
| [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)     | [11.2.0.252](https://docs.nvidia.com/cuda/cublas/index.html) | [11.2.0.252](https://docs.nvidia.com/cuda/cublas/index.html) |
| [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html) | [8.0.4](https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html) | [8.0.4](https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html) |
| [NCCL](https://docs.nvidia.com/deeplearning/nccl/archives/index.html) | [2.7.8](https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html) | [2.7.8](https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html) |
| **Packages**                                                 |                                                              |                                                              |
| [CMake](https://cmake.org/)                                  | \<\<[3.18.2](https://cmake.org/cmake/help/latest/release/3.18.html)\>\> | [3.17.0](https://cmake.org/cmake/help/latest/release/3.17.html) |
| [hwloc](https://www.open-mpi.org/projects/hwloc/)            | \<\<[2.4.0](https://www.open-mpi.org/projects/hwloc/doc/)\>\> | [2.2.0](https://www.open-mpi.org/projects/hwloc/doc/)        |
| [UCX](https://www.openucx.org/)                              | \<\<[1.9.0](https://github.com/openucx/ucx/releases/tag/v1.9.0)>> | [1.8.0](https://github.com/openucx/ucx/releases/tag/v1.8.0)  |
| [OpenMPI](https://www.open-mpi.org/)                         | \<\<[4.1.0](https://www.open-mpi.org/software/ompi/v4.1/)>\> | [4.0.3](https://www.open-mpi.org/software/ompi/v4.0/)        |
| [rapids/rmm](https://htmlpreview.github.io/?https://github.com/XiaoleiShi-NV/HugeCTR/blob/master/docs/software_stack_src/index.html#libraries) | \<\<[0.17](https://github.com/rapidsai/rmm/releases/tag/v0.17.0)>> | [0.16](https://github.com/rapidsai/rmm/releases/tag/v0.16.0) |
| [rapids/cudf](https://htmlpreview.github.io/?https://github.com/XiaoleiShi-NV/HugeCTR/blob/master/docs/software_stack_src/index.html#libraries) | \<\<[0.17](https://github.com/rapidsai/cudf/releases/tag/v0.17.0)>> | [0.16](https://github.com/rapidsai/cudf/releases/tag/v0.16.0) |
| [TensorFlow](https://www.tensorflow.org/)                    | \<\<[2.4.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.0)>> | nightly-gpu                                                  |
| Python libs                                                  | numpy<br>pandas<br>sklearn<br>ortools<br>jupyter             | numpy<br>pandas<br>sklearn<br>ortools<br>jupyter             |

<br></br>

<div align="center"><a name="md_cap_plugin">Table 2. Software stack for embedding TF plugin image (plugin-embedding.Dockerfile)</a></div>

| Container Image                                              | v3.0                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **DGX**                                                      |                                                              |
| DGX System                                                   | DGX-1<br>DGX-2<br>DGX A100<br>DGX Station                    |
| **NVIDIA Certified Systems**                                 |                                                              |
| [NVIDIA Driver](http://www.nvidia.com/Download/index.aspx?lang=en-us) | Release v3.0 is based on CUDA 11.0.221, which requires NVIDIA driver release 450.51.<br><br>However, if you are running on Tesla (for example, T4 or any other Tesla board), you may use NVIDIA driver release 418.xx or 440.30.<br><br>The CUDA driver's compatibility package only supports particular drivers. <a href="#md_idx_1">1</a> |
| GPU Model                                                    | [NVIDIA Ampere GPU Architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture)<br>[Turing](https://www.nvidia.com/en-us/geforce/turing/)<br>[Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)<br>[Pascal](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) |
| **Base Container Image**                                     |                                                              |
| Container OS                                                 | [Ubuntu 18.04](http://releases.ubuntu.com/18.04/)            |
| [CUDA](http://docs.nvidia.com/cuda/index.html)               | [11.0.221](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) |
| [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)     | [11.2.0.252](https://docs.nvidia.com/cuda/cublas/index.html) |
| [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html) | [8.0.4](https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html) |
| [NCCL](https://docs.nvidia.com/deeplearning/nccl/archives/index.html) | [2.7.8](https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html) |
| **Packages**                                                 |                                                              |
| [CMake](https://cmake.org/)                                  | [3.18.4](https://cmake.org/cmake/help/latest/release/3.18.html) |
| [rapids/rmm](https://htmlpreview.github.io/?https://github.com/XiaoleiShi-NV/HugeCTR/blob/master/docs/software_stack_src/index.html#libraries) | [0.17](https://github.com/rapidsai/rmm/releases/tag/v0.17.0) |
| [TensorFlow](https://www.tensorflow.org/)                    | [2.4.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.0) |
| Python libs                                                  | numpy<br>pandas<br>sklearn<br>ortools<br>jupyter<br>nvtx-plugins |

<a name="md_idx_1">1</a> For a complete list of supported drivers, see the [CUDA Application Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#cuda-application-compatibility) topic. For more information, see [CUDA Compatibility and Upgrades](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#cuda-compatibility-and-upgrades).

* * *
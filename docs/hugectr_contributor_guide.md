# Contributing to HugeCTR

We are grateful for your interest in HugeCTR. You can contribute to HugeCTR in three ways:
1. Report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/NVIDIA/HugeCTR/issues/new/choose) describing issues you face or features you'd like to add to HugeCTR.
    - After the HugeCTR team's thorough review, they will be assigned to a future release. If you think the issue should be prioritized over others, comment on the issue .
2. Prose and implement a feature yourself.
    - Post your detailed proposal, so that we can discuss how to design and implementation it.
    - Once we agree upon its plan and design, go ahead and implement it, using the [Code Contribution Guide](#code-contribution-guide) below.
3. Implement our pending feature or fix an outstanding bug
    - Follow the [Code Contribution Guide](#code-contribution-guide) guide below.
    - If you need more information on the particular issue, discuss with us via comments on the issue.

## Code Contribution Guide

1. Follow the [setup development environment](#setup-development-environment) guide below to learn how to build the HugeCTR or Sparse Operation Kit from the source code
2. [File an issue](https://github.com/NVIDIA/HugeCTR/issues/new/choose), and comment that you will work on it.
3. Code! Don't forget to add or update unit tests properly!
4. When done, [create your pull request](https://github.com/nvidia/HugeCTR/compare)
5. Wait for a maintainer to review your code; you may be asked to update your code if necessary.
6. Once reviewed and approved, a maintainer will merge your pull request

If you have any questions or need clarifications during your development, do not hesitate to contact us via comments.
Thank you for your contirbution to improivng HugeCTR!

## Setup Development Environment
### Build HugeCTR from source code
To build HugeCTR from the source code, here are the steps to follow:

* Build the `hugectr:devel` image by following [Build Container for Model Training](../tools/dockerfiles#build-container-for-model-training).
Please choose the **Development mode** in this case.

* Download the HugeCTR repository and the third-party modules that it relies on by running the following commands:
  ```shell
  $ git clone https://github.com/NVIDIA/HugeCTR.git
  $ cd HugeCTR
  $ git submodule update --init --recursive
  ```
* Build HugeCTR from scratch using one or any combination of the following options:
  - **SM**: You can use this option to build HugeCTR with a specific compute capability (DSM=80) or multiple compute capabilities (DSM="70;75"). The default compute capability is 70, which uses the NVIDIA V100 GPU. For more information, see [Compute Capability](#compute-capability). 60 is not supported for inference deployments. For more information, see [Quick Start](https://github.com/triton-inference-server/hugectr_backend#quick-start).
  - **CMAKE_BUILD_TYPE**: You can use this option to build HugeCTR with Debug or Release. When using Debug to build, HugeCTR will print more verbose logs and execute GPU tasks in a synchronous manner.
  - **VAL_MODE**: You can use this option to build HugeCTR in validation mode, which was designed for framework validation. In this mode, loss of training will be shown as the average of eval_batches results. Only one thread and chunk will be used in the data reader. Performance will be lower when in validation mode. This option is set to OFF by default.
  - **ENABLE_MULTINODES**: You can use this option to build HugeCTR with multi-nodes. This option is set to OFF by default. For more information, see [samples/dcn2nodes](../samples/dcn).
  - **ENABLE_INFERENCE**: You can use this option to build HugeCTR in inference mode, which was designed for the inference framework. In this mode, an inference shared library will be built for the HugeCTR Backend. Only interfaces that support the HugeCTR Backend can be used. Therefore, you can’t train models in this mode. This option is set to OFF by default.

  Here are some examples of how you can build HugeCTR using these build options:
  ```shell
  $ mkdir -p build && cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Target is NVIDIA V100 with all others default
  $ make -j && make install
  ```

  ```shell
  $ mkdir -p build && cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70,80" -DVAL_MODE=ON .. # Target is NVIDIA V100 / A100 and Validation mode on.
  $ make -j && make install
  ```

  ```shell
  $ mkdir -p build && cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70,80" -DCMAKE_BUILD_TYPE=Debug .. # Target is NVIDIA V100 / A100, Debug mode.
  $ make -j && make install
  ```

  ```shell
  $ mkdir -p build && cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70,80" -DENABLE_INFERENCE=ON .. # Target is NVIDIA V100 / A100 and Validation mode on.
  $ make -j && make install
  ```

### Build Sparse Operation Kit (SOK) from source code
To build the Sparse Operation Kit component in HugeCTR, here are the steps to follow:
* Build the `hugectr:tf-plugin` docker image by following [Build container for TF plugin](../tools/dockerfiles/README.md#build-container-for-tensorflow-plugin).
Please chose the **Development mode** in this case.
* Download the HugeCTR repository by running the following commands:
  ```shell
  $ git clone https://github.com/NVIDIA/HugeCTR.git hugectr
  ```
* Build and install libraries to the systam paths.
  ```shell
  $ cd hugectr/sparse_operation_kit
  $ bash ./install.sh --SM=[GPU Compute Capability] --USE_NVTX=[ON/OFF]
  ```
  If you want to profiling this module with nvtx, you can enable nvtx marks by setting `--USE_NVTX=ON`.

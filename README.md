# <img src="docs/user_guide_src/merlin_logo.png" alt="logo" width="85"/> Merlin: HugeCTR #
[![v23](docs/user_guide_src/version.JPG)](docs/hugectr_user_guide.md/#whats-new-in-version-23)

HugeCTR is a recommender specific framework which is capable of distributed training across multiple GPUs and nodes for Click-Through-Rate (CTR) estimation.
It is the training component of [**NVIDIA Merlin**](https://developer.nvidia.com/nvidia-merlin#getstarted),
which is an open beta framework accelerating the entire pipeline from data ingestion and training to deploying GPU-accelerated recommender systems.
This document contains the instructions on how to quickly get started with HugeCTR, including the use of our docker image and one of our samples.
For more detailed information, please refer to [**HugeCTR User Guide**](docs/hugectr_user_guide.md). 

Design Goals:
* Fast: HugeCTR is a speed-of-light CTR model framework.
* Dedicated: HugeCTR focuses on essential things which you need in training your CTR model.
* Easy: you can quickly be used to HugeCTR, regardless of whether you are a data scientist or machine learning practitioner. 

## Version 2.3
HugeCTR version 2.3 includes the features which enrich interoperability and user convenience such as Python Interface, HugeCTR embedding as Tensorflow Op, Model Prefetching. You can also find the full list of changes [**here**](docs/hugectr_user_guide.md#whats-new-in-version-23).

## Getting Started with NGC
A production docker image of HugeCTR is available in the NVIDIA container repository at the following location: https://ngc.nvidia.com/catalog/containers/nvidia:hugectr.

You can pull and launch the container using the following command:
```shell
docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) nvcr.io/nvidia/hugectr:v2.3
# To use Tensorflow, especially if you want to try the HugeCTR embedding op, please use tag `v2.3_tf` instead.
```
If you are running on a docker version 19+, change `--runtime=nvidia` to `--gpus all`.

This image contains the executable files only enough for production use cases. For the full installation, please refer to [the quick start section](quick-start).

## Quick Start

### 1. Download Repository ###
You can download the HugeCTR repository and the third party modules which it relies upon:
```shell
git clone https://github.com/NVIDIA/HugeCTR.git
cd HugeCTR
git submodule update --init --recursive
```
###  2. Build Docker Image and HugeCTR ###
Inside the HugeCTR directory, build a docker image:
```shell
# It may take a long time to download and build all the dependencies.
docker build -t hugectr:devel -f ./tools/dockerfiles/dev.Dockerfile . # To use Tensorflow, especially if you want to try our HugeCTR embedding op with Tensorflow, please use dev.tfplugin.Dockerfile instead.
```
If you have Docker 19.03 or later, run the container with the following command: 
```shell
docker run --gpus all --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr hugectr:devel
```
Otherwise, run the container with the following command: 
```shell
docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr hugectr:devel
```


Then build HugeCTR with the build type and compute capability specified. Please see [**Build Options**](#build-options) for more details.

First, make a build directory and get inside it:
```shell
mkdir -p build
cd build
```
If your target graphic card is NVIDIA A100, run the following command. If you are using NVIDIA V100 or NVIDIA T4, change `-DSM=80` to `-DSM=70` or `-DSM=75`.
```shell
cmake -DCMAKE_BUILD_TYPE=Release -DSM=80 .. # Target is NVIDIA A100
```
Finally build it. HugeCTR executable file and unit tests are located inside `build/bin`.
```shell
make -j
```

###  3. Download and Preprocess Dataset ###
Let’s download [the Kaggle Display Advertising Challenge Dataset](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset) to `$HugeCTR/tools/criteo_script/` and preprocess it to train [the Deep & Cross Network](https://arxiv.org/pdf/1708.05123.pdf) (DCN) example:
```shell
cd ../../tools/criteo_script/ # assume that the downloaded dataset is here
bash preprocess.sh dcn 1 0
```

Alternatively you can generate a synthetic dataset:
```shell
cd /hugectr/build
mkdir dataset_dir
bin/data_generator ../samples/dcn/dcn.json ./dataset_dir 434428 1
```

### 4. Train an example DCN model ###
```shell
cd /hugectr/build
bin/huge_ctr --train ../samples/dcn/dcn.json
```
The other sample models and their end-to-end instructions are available inside [this directory](/samples).

## Table of Contents
* [Requirements](#requirements)
* [**Supported Compute Capabilities**](#supported-compute-capabilities)
* [Build Options](#build-options)
* [Synthetic Data Generation and Benchmark](#synthetic-data-generation-and-benchmark)
* [File Format](#file-format)
* [Document Generation](#document-generation)
* [Coding Style and Refactor](#coding-style-and-refactor)

## Requirements ##
* cuBLAS >= 10.1
* Compute Capability >= 70 (V100)
* CMake >= 3.17.0
* cuDNN >= 7.5
* NCCL >= 2.0
* RMM >= 0.16
* CUDF >= 0.16
* Clang-Format 3.8
* GCC >= 7.4.0
* ortools >= 7.6.7691
### Optional, if require multi-nodes training ###
* OpenMPI >= 4.0
* UCX library >= 1.8.0
* HWLOC library >= 2.2.0
* mpi4py >= 3.0.3

## Supported Compute Capabilities ##
|Compute Compatibility|GPU|
|----|----|
|60|NVIDIA P100 (Pascal)|
|70|NVIDIA V100 (Volta)|
|75|NVIDIA T4 (Turing)|
|80|NVIDIA A100 (Ampere)|

## Build Options ##
### Use Docker Container ###
You can choose to use Docker to simplify the environment setting up.
Mare sure that you have installed [**Nvidia Docker**](https://github.com/NVIDIA/nvidia-docker) .

To build a docker image of **development environment** from the corresponding Dockerfile, run the command below.
It will install the libraries and tools required to use HugeCTR.
HugeCTR build itself must be done by yourself.
```shell
$ docker build -t hugectr:devel -f ./tools/dockerfiles/dev.a100.Dockerfile .
```
Run with interaction mode (mount the home directory of repo into container for easy development and try):
```shell
$ docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr hugectr:devel
```

To build a docker image of **production environment**, run the command below.
In addition to resolving dependencies, it will build and install HugeCTR to `/usr/local/hugectr`.
Note that `SM` (the target GPU architecture list) and `NCCL_A2A` (use NCCL ALL-to-ALL or not) are also specified.
You can change them according to your environment.
```shell
$ docker build --build-arg SM="70;75;80" \
               --build-arg NCCL_A2A=on \
               -t hugectr:build \
               -f ./tools/dockerfiles/build.Dockerfile .
```

### Build with Release ###
Compute Capability can be specified by `-DSM=[Compute Compatibilities]`, which is SM70 by default (NVIDIA V100). It is also possible to set multiple Compute Capabilities, e.g., `-DSM=70` for NVIDIA V100 and `-DSM="70;75"` for both NVIDIA V100 and NVIDIA T4.
```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Target is NVIDIA V100
$ make -j
```

### Build with Debug ###
If the build type is `Debug`, HugeCTR will print more verbose logs and execute GPU tasks in a synchronous manner.
The other options remain the same as `Release` build.
```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Debug -DSM=70 .. # Target is NVIDIA V100
$ make -j
```

### Build with Validation Mode ###
This mode is designed for framework validation. In this mode loss of training will be shown as the average of `eval_batches` results. Only one thread and chunk will be used in DataReader. Performance will be lower than turning off.
```shell
$ mkdir -p build
$ cd build
$ cmake -DVAL_MODE=ON ..
$ make -j
```

### Build with Multi-Node Training Supported ###
To run with multi-nodes please build in this way and run HugeCTR with `mpirun`. For more details please refer to `samples/dcn2nodes`
```shell
$ mkdir -p build
$ cd build
$ cmake -DENABLE_MULTINODES=ON ..
$ make -j
```

### Build with NCCL All2All Supported ###
The default collection communication library used in LocalizedSlotSparseEmbedding is [Gossip](https://github.com/Funatiq/gossip). [NCCL all2all](https://github.com/NVIDIA/nccl/tree/p2p) is also supported in HugeCTR. If you want to run with NCCL all2all, please turn on the NCCL_A2A switch in cmake. 
```shell
$ mkdir -p build
$ cd build
$ cmake -DNCCL_A2A=ON ..
$ make -j
```

## Synthetic Data Generation and Benchmark ##
For quick benchmarking and research use, you can generate a synthetic dataset like below. Without any additional modification to JSON file. Both [**Norm** format](#norm) (with Header) and [**Raw** format](#raw) (without Header) dataset can be generated with `data_generator`. For categorical features, you can configure the probability distribution to be uniform or power-law.
The default distribution is uniform.
- For `Norm` format: <br>
```bash
$ ./data_generator your_config.json data_folder vocabulary_size max_nnz (--files <number_of_files>) (--samples <num_samples_per_file>) (--long-tail <long|short|medium>)
$ ./huge_ctr --train your_config.json
```
- For `Raw` format: <br>
```bash
$ ./data_generator your_config.json (--long-tail <long|medium|short>)
$ ./huge_ctr --train your_config.json
```

Parameters:
+ `data_folder`: Directory where the generated dataset is stored.
+ `vocabulary_size`: Total vocabulary size of your target dataset, which cannot be exceed `max_vocabulary_size_per_gpu` **x** the number of active GPUs. 
+ `max_nnz`: You can use this parameter to simulate one-/multi-hot encodings. If you just want to use the one-hot encoding, set this parameter to 1. Otherwise, [1, max_nnz] values will be generated for each slot. **Note** that `max_nnz * slot_num` must be less than `max_feature_num_per_sample` in the data layer of the used JSON config file.
+ `--files`: Number of data files will be generated (optional). The default value is `128`.
+ `--samples`: Number of samples per file (optional). The default value is `40960`.
+ `--long-tail`: If you want to generate data with power-law distribution for categorical features, you can use this option. There are three option values to be chosen from, i.e., `long`, `medium` and `short`, which characterize the properties of the tail. The scaling exponent will be 1, 3, and 5 correspondingly.

## Coding Style and Refactor ##
Default coding style follows Google C++ coding style [(link)](https://google.github.io/styleguide/cppguide.html).
This project also uses `Clang-Format`[(link)](https://clang.llvm.org/docs/ClangFormat.html) to help developers to fix style issue, such as indent, number of spaces to tab, etc.
The Clang-Format is a tool that can auto-refactor source code.
Use following instructions to install and enable Clang-Format:
### Install ###
```shell
$ sudo apt-get install clang-format
```
### Run ###
```shell
# First, configure Cmake as usual 
$ mkdir -p build
$ cd build
$ cmake -DCLANGFORMAT=ON ..
# Second, run Clang-Format
$ cmake --build . --target clangformat
# Third, check what Clang-Format did modify
$ git status
# or
$ git diff
```

## Document Generation ##
You can consider to use Doxygen if you are interested in understanding HugeCTR's C++ class hierarchy and their functions.
By default, inside the  `docs` directory, a HTML document and a LaTeX based reference manual are generated.

### Install ###
[Download doxygen](http://www.doxygen.nl/download.html)
### Generation ###
Within project `home` directory
```shell
$ doxygen
```

## Contributing ##
Merlin HugeCTR is an industry oriented framework and we are keen on making sure that it satisfies your needs and that it provides an overall pleasant experience.
If you face any problem or have any question, please file an issue [here](https://github.com/NVIDIA/HugeCTR/issues) so that we can discuss it together.
We would also be grateful if you have any suggestions or feature requests to enrich HugeCTR. 
To further advance the Merlin/HugeCTR Roadmap, we would like to invite you to share the gist of your recommender system pipeline via [this survey](https://developer.nvidia.com/merlin-devzone-survey)


# HugeCTR A100 PREVIEW #
[![v21](docs/user_guide_src/v21.JPG)](docs/hugectr_user_guide.md#new-features-in-version-21)

To demonstrate the performance of HugeCTR on A100, we introduce this preview version. We support new features like full fp16 pipeline / algorithm search / AUC calculation and many of these features will be available in the next release soon. 

## Intention ##
HugeCTR is a high-efficiency GPU framework designed for Click-Through-Rate (CTR) estimation training, and the new released NVIDIA A100 GPU has excellent acceleration on various scales for AI, data analysis and high performance computing (HPC), and meet extremely severe computing challenges. To demonstrate HugeCTR’s performance on A100 GPU, this version is developed to leverage new features of the latest GPU.

In order to get better performance and flexibility, new features have been added in this version.

## New features ##
+ **Algorithm Search** : Support algorithm selection in fully connected layers for better performance.

+ **AUC** : Support AUC calculation for accuracy evaluation.

+ **Batch shuffle and last batch in eval** : Support batch shuffle and the last batch during 	evaluation won’t be dropped.

+ **Different batch size in training and evaluation** : Support this for best performance in evaluation.

+ **Full FP16 pipeline** : In order to be able to process more data simultaneously and obtain better performance, Full FP16 pipeline is supported in this version.

+ **Fused fully connected layer** : Fused bias adding and relu activation into a single layer.

+ **Caching evaluation data on device** : For the GPUs with large memory like A100, we can use caching data for small evaluation data sets.

+ **Interaction layer** : Support this famous layer used in CTR estimation.

+ **Optimized data reader for raw format** : Each sample has 40 32bits integers, where the first integer is label, the next 13 integers are dense feature and the following 26 integers are category feature.

+ **Deep Learning Recommendation Model (DLRM)** : DLRM support please find more details in [samples/dlrm](samples/dlrm/README.md).

+ **Learning rate scheduling** : Support different learning rate scheduling. <br>
<div align=center><img width = '500' src ="docs/user_guide_src/learning_rate_scheduling.png"/></div>
<div align=center>Fig 1. Learning rate scheduling</div>

 
# HugeCTR #

HugeCTR is a high-efficiency GPU framework designed for Click-Through-Rate (CTR) estimation training.

Design Goals:
* Fast: it's a speed-of-light CTR training framework;
* Dedicated: we consider everything you need in CTR training;
* Easy: you can start your work now, no matter you are a data scientist, a learner, or a developer.

Please find more introductions in our [**HugeCTR User Guide**](docs/hugectr_user_guide.md) and [**Questions and Answers**](docs/QAList.md) in directory `docs/`


## Requirements ##
* cuBLAS >= 9.1
* Compute Capability >= 60 (P100)
* CMake >= 3.8
* cuDNN >= 7.5
* NCCL >= 2.0
* Clang-Format 3.8
* GCC >= 7.4.0
* ortools
### Optional, if require multi-nodes training ###
* OpenMPI >= 4.0
* UCX library >= 1.6
* HWLOC library >= 2.1.0
* mpi4py

## Build ##
### Init Git ###
Under the home directory of HugeCTR:
```shell
$ git submodule update --init --recursive
```

### Use Docker Container ###
You can choose using docker to simplify the environment setting up, otherwise please jump to the next paragraph directly.

Ensure that you have [**Nvidia Docker**](https://github.com/NVIDIA/nvidia-docker) installed.

To build docker image from the Dockerfile, run the command:
```shell
$ docker build -t hugectr:devel .
```

After building the docker image, for ease of use, you can push it to your docker registry

Now, you can enter the development environment by running a HugeCTR docker container, you need to mount your dataset into docker container as well
```shell
$ docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr hugectr:devel bash
```

Then continue with the following steps

### Build with Release ###
Compute Capability can be specified by `-DSM=[Compute Compatibilities]`, which is SM70 by default (Tesla P100). One or more Compute Capabilities are avaliable to be set. E.g. `-DSM=70` for Telsa V100 and `-DSM="70;75"` for both Telsa V100 and Telsa T4.
```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. #using Tesla V100
$ make -j
```

Supported Compatibility and Tesla GPUs:

|Compute Compatibility|GPU|
|----|----|
|70|Tesla V100|
|75|Tesla T4|
|80|Tesla A100|

### Build with Debug ###
Compute Capability can be specified by `-DSM=[Compute Compatibilities]`, which is SM70 by default (Tesla P100). One or more Compute Capabilities are avaliable to be set. E.g. `-DSM=70` for Telsa V100 and `-DSM="70;75"` for both Telsa V100 and Telsa T4.
```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Debug -DSM=70 .. #using Telsa V100
$ make -j
```

### Build with Validation Mode ###
This mode is designed for framework validation. In this mode loss of trainig will be shown as the average of `eval_batches` results. Only one thread and chunk will be used in DataReader. Performance will be lower than turning off.
```shell
$ mkdir -p build
$ cd build
$ cmake -DVAL_MODE=ON ..
$ make -j
```

### Build with Multi-Nodes Training Supported ###
To run with multi-nodes please build in this way and run HugeCTR with `mpirun`. For more details plese refer to `samples/dcn2nodes`
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

## Run ##
Please refer to samples/*

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
Doxygen is supported in HugeCTR and by default on-line documentation browser (in HTML) and an off-line reference manual (in LaTeX) can be generated within `docs/`.
### Install ###
[Download doxygen](http://www.doxygen.nl/download.html)
### Generation ###
Within project `home` directory
```shell
$ doxygen
```

## Benchmark ##
Random data set can be generated according to your JSON network config file (`your_config.json`) with `data_generator` for easy benchmark. Usage:
```shell
$ ./data_generator your_config.json data_folder vocabulary_size max_nnz [option:#files] [option:#samples per file]
$ ./huge_ctr --train your_config.json
```
Arguments:
* `data_folder`: You have to specify the folder for the generated data
* `vocabulary_size`: Vocabulary size of your target data set
* `max_nnz`: [1,max_nnz] values will be generated for each feature (slot) in the data set. Note that max_nnz * #slot should be less than the `max_feature_num` in your data layer.
* `#files`: number of data file will be generated.
* `#samples per file`: number of samples per file. 


## File Format ##
Totally three kinds of files will be used as input of HugeCTR Training: configuration file (.json), model file, data set.

### Configuration File ###
Configuration file should be a json format file e.g. [simple_sparse_embedding.json](utest/session/simple_sparse_embedding.json)

There are four sessions in a configuration file: "solver", "optimizer", "data", "layers". The sequence of these sessions is not restricted.
* You can specify the device (or devices), batchsize, model_file.. in `solver` session;
* and the `optimizer` that will be used in every layer.
* Finally, layers should be listed under `layers`. Note that embedders should always be the first layers.

### Model File ###
Model file is a binary file that will be loaded for weight initilization.
In model file weight will be stored in the order of layers in configuration file. 

[**Here**](./tutorial/dump_to_tf/readMe.md) we provide a tutorial of ```dumping models to TensorFlow```, and explain more details about the ```model format```.

### Data Set ###
A data set includes a ASCII format file list and a set of data in binary format.

A file list starts with a number which indicate the number of files in the file list, then comes with the path of each data file.
```shell
$ cat simple_sparse_embedding_file_list.txt
10
./simple_sparse_embedding/simple_sparse_embedding0.data
./simple_sparse_embedding/simple_sparse_embedding1.data
./simple_sparse_embedding/simple_sparse_embedding2.data
./simple_sparse_embedding/simple_sparse_embedding3.data
./simple_sparse_embedding/simple_sparse_embedding4.data
./simple_sparse_embedding/simple_sparse_embedding5.data
./simple_sparse_embedding/simple_sparse_embedding6.data
./simple_sparse_embedding/simple_sparse_embedding7.data
./simple_sparse_embedding/simple_sparse_embedding8.data
./simple_sparse_embedding/simple_sparse_embedding9.data
```

A data file (binary) contains a header and data (many samples). 

Header Definition:
```c
typedef struct DataSetHeader_ {
  long long error_check;        // 0: no error check; 1: check_num
  long long number_of_records;  // the number of samples in this data file
  long long label_dim;          // dimension of label
  long long dense_dim;          // dimension of dense feature
  long long slot_num;           // slot_num for each embedding
  long long reserved[3];        // reserved for future use
} DataSetHeader;

```

Data Definition (each sample):
```c
typedef struct Data_{
  int length;                   // bytes in this sample (optional: only in check_sum mode )
  float label[label_dim];       
  float dense[dense_dim];
  Slot slots[slot_num];          
  char checkbits;                // checkbit for this sample (optional: only in checksum mode)
} Data;

typedef struct Slot_{
  int nnz;
  long long*  keys; 
} Slot;
```

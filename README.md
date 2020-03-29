# HugeCTR #
[![v21](docs/user_guide_src/v21.JPG)](docs/hugectr_user_guide.md#new-features-in-version-21)

HugeCTR is a high-efficiency GPU framework designed for Click-Through-Rate (CTR) estimation training.

Design Goals:
* Fast: it's a speed-of-light CTR training framework;
* Dedicated: we consider everything you need in CTR training;
* Easy: you can start your work now, no matter you are a data scientist, a learner, or a developer.

Please find more introductions in our [**HugeCTR User Guide**](docs/hugectr_user_guide.md) and doxygen files in directory `docs/`

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
### Building HugeCTR in docker container ###
For a development environment where can build HugeCTR, you can use the provided Dockerfile

To build docker image from the Dockerfile, run the command:

```shell
$ docker build -t hugectr:latest .
```

After building the docker image, you can enter the development environment by running a docker container

```shell
$ docker run --runtime=nvidia -it hugectr:latest bash
```

Then continue with the following steps

### Init Git ###
```shell
$ git submodule update --init --recursive
```

### Build with Release ###
Compute Capability can be specified by `-DSM=XX`, which is SM=60 by default. Only one Compute Capability is avaliable to be set.
```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM=XX ..
$ make
```

### Build with Debug ###
Compute Capability can be specified by `-DSM=XX`, which is SM=60 by default. Only one Compute Capability is avaliable to be set.
```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Debug -DSM=XX ..
$ make
```

### Build with Validation Mode ###
This mode is designed for framework validation. In this mode loss of trainig will be shown as the average of `eval_batches` results. Only one thread and chunk will be used in DataReader. Performance will be lower than turning off.
```shell
$ mkdir -p build
$ cd build
$ cmake -DVAL_MODE=ON ..
$ make
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

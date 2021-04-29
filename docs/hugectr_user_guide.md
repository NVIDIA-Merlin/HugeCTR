HugeCTR User Guide
==================

HugeCTR is a GPU-accelerated framework designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs). HugeCTR supports model-parallel embedding tables and data-parallel neural networks and their variants such as [Wide and Deep Learning (WDL)](https://arxiv.org/abs/1606.07792), [Deep Cross Network (DCN)](https://arxiv.org/abs/1708.05123), [DeepFM](https://arxiv.org/abs/1703.04247), and [Deep Learning Recommendation Model (DLRM)](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/). HugeCTR is a component of [NVIDIA Merlin Open Beta](https://developer.nvidia.com/nvidia-merlin#getstarted), used to build large-scale deep learning recommender systems.

<div align=center><img src ="user_guide_src/merlin_arch.png"/></div>
<div align=center>Fig. 1: Merlin Architecture</div>

<br></br>

To prevent data loading from becoming a major bottleneck during training, HugeCTR contains a dedicated data reader that is inherently asynchronous and multi-threaded. It will read a batched set of data records in which each record consists of high-dimensional, extremely sparse (or categorical) features. Each record can also include dense numerical features, which can be fed directly to the fully connected layers. An embedding layer is used to compress the input-sparse features to lower-dimensional, dense-embedding vectors. There are three GPU-accelerated embedding stages:
* table lookup
* weight reduction within each slot
* weight concatenation across the slots

To enable large embedding training, the embedding table in HugeCTR is model parallel and distributed across all GPUs in a homogeneous cluster, which consists of multiple nodes. Each GPU has its own:
* feed-forward neural network (data parallelism) to estimate CTRs
* hash table to make the data preprocessing easier and enable dynamic insertion

Embedding initialization is not required before training since the input training data are hash values (64bit long long type) instead of original indices. A pair of <key,value> (random small weight) will be inserted during runtime only when a new key appears in the training data and the hash table cannot find it.

<div align=center><img src="user_guide_src/fig1_hugectr_arch.png" width="781" height="333"/></div>
<div align=center>Fig. 2: HugeCTR Architecture</div>

<br></br>

<div align=center><img src="user_guide_src/fig2_embedding_mlp.png" width="389" height="244"/></div>
<div align=center>Fig. 3: Embedding Architecture</div>

<br></br>

<div align=center><img src="user_guide_src/fig3_embedding_mech.png" width="502" height="225" /></div>
<div align=center>Fig. 4: Embedding Mechanism</div>

<br></br>

## Table of Contents
* [Installing and Building HugeCTR](#installing-and-building-hugectr)
* [Use Cases](#use-cases)
* [Core Features](#core-features)
* [Tools](#tools)

## Installing and Building HugeCTR ##
You can either install HugeCTR easily using the Merlin Docker image in NGC, or build HugeCTR from scratch using various build options if you're an advanced user.

### Compute Capability ###
We support the following compute capabilities:

| Compute Capability | GPU                  | [SM](#building-hugectr-from-scratch) |
|--------------------|----------------------|----|
| 6.0                | NVIDIA P100 (Pascal) | 60 |
| 7.0                | NVIDIA V100 (Volta)  | 70 |
| 7.5                | NVIDIA T4 (Turing)   | 75 |
| 8.0                | NVIDIA A100 (Ampere) | 80 |

### Software Stack ###
Please find the detailed software stack (dependencies) of HugeCTR under this [link](../tools/dockerfiles/software_stack.md).

### Installing HugeCTR from NGC Containers
All NVIDIA Merlin components are available as open source projects. However, a more convenient way to make use of these components is by using our Merlin NGC containers. Containers allow you to package your software application, libraries, dependencies, and runtime compilers in a self-contained environment. When installing HugeCTR using NGC containers, the application environment remains portable, consistent, reproducible, and agnostic to the underlying host system software configuration.

HugeCTR is included in the Merlin Docker image, which is available in the NVIDIA container repository on https://ngc.nvidia.com/catalog/containers/nvidia:hugectr.

You can pull and launch the container by running the following command:
```shell
$ docker run --runtime=nvidia --rm -it nvcr.io/nvidia/merlin/merlin-training:0.5  # Start interaction mode
``` 

Activate the merlin conda environment by running the following command:  
```shell.
source activate merlin
```  

### Building Your Own Container ###
Please refer to [Build HugeCTR Docker Containers](../../tools/dockerfiles) to build the HugeCTR Docker image on your own.

### Building HugeCTR from Scratch
Before building HugeCTR from scratch, you should prepare the dependencies according to [link](../../tools/dockerfiles/software_stack.md). Then download the HugeCTR repository and the third-party modules that it relies on by running the following commands:
```shell
$ git clone https://github.com/NVIDIA/HugeCTR.git
$ cd HugeCTR
$ git submodule update --init --recursive
```

You can build HugeCTR from scratch using one or any combination of the following options:
* **SM**: You can use this option to build HugeCTR with a specific compute capability (DSM=80) or multiple compute capabilities (DSM="70;75"). The default compute capability is 70, which uses the NVIDIA V100 GPU. See [Compute Capability](#compute-capability) for more detailed information. 60 is not supported for inference deployments. For more information, see [Quick Start](https://github.com/triton-inference-server/hugectr_backend#quick-start).
* **CMAKE_BUILD_TYPE**: You can use this option to build HugeCTR with Debug or Release. When using Debug to build, HugeCTR will print more verbose logs and execute GPU tasks in a synchronous manner.
* **VAL_MODE**: You can use this option to build HugeCTR in validation mode, which was designed for framework validation. In this mode, loss of training will be shown as the average of eval_batches results. Only one thread and chunk will be used in the data reader. Performance will be lower when in validation mode. This option is set to OFF by default.
* **ENABLE_MULTINODES**: You can use this option to build HugeCTR with multi-nodes. This option is set to OFF by default. For more information, see [samples/dcn2nodes](../samples/dcn2nodes).
* **NCCL_A2A**: You can use this option to build HugeCTR with NCCL All2All, which is the default collection communication library used in LocalizedSlotSparseEmbedding. Gossip is also supported in HugeCTR, which provides better performance on servers without NVSwitch support. To build HugeCTR with NCCL All2All, please turn on the NCCL_A2A switch in the cmake. This option is set to OFF by default.
* **ENABLE_INFERENCE**: You can use this option to build HugeCTR in inference mode, which was designed for the inference framework. In this mode, an inference shared library will be built for the HugeCTR Backend. Only interfaces that support the HugeCTR Backend can be used. Therefore, you can’t train models in this mode. This option is set to OFF by default.

Here are some examples of how you can build HugeCTR using these build options:
```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Target is NVIDIA V100 with all others default
$ make -j
```

```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70,80" -DVAL_MODE=ON .. # Target is NVIDIA V100 / A100 and Validation mode on.
$ make -j
```

```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70,80" -DCMAKE_BUILD_TYPE=Debug -DNCCL_A2A=OFF .. # Target is NVIDIA V100 / A100, Debug mode and Gossip for all2all data transaction.
$ make -j
```

```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70,80" -DENABLE_INFERENCE=ON .. # Target is NVIDIA V100 / A100 and Validation mode on.
$ make -j
```

## Use Cases ##
The Python interface can be used to quickly and easily train models while the C++ interface can be used to train with one-hot/multi-hot data.

### Training Models with the Python Interface
If you are already using a configuration file to train models on HugeCTR, you'll only have to locate the `hugectr.so` file when training models using the Python interface. For additional information, see [Configuration File Setup](./configuration_file_setup.md).

You'll also need to set the `PYTHONPATH` environment variable. You can still configure your model in your configuration file, but the training options such as `batch_size` must be specified through `hugectr.solver_parser_helper()` in Python. For additional information regarding how to use the HugeCTR Python API and comprehend its API signature, see our [Jupyter Notebook Tutorial](../notebooks/python_interface.ipynb).

### Training One-Hot and Multi-Hot Data with the C++ Interface
If training with a single node using the C++ interface, run the following command:
```shell
$ huge_ctr --train <config>.json
```

You'll need to create a configuration file in order to train with one-hot and multi-hot data. To load a particular snapshot, modify the `dense_model_file` and `sparse_model_file` files within the solver clause for that snapshot. For additional information, see [Configuration File Setup](./configuration_file_setup.md) and [samples](../samples).

## Core Features ##
In addition to single node and full precision training, HugeCTR supports a variety of features including the following:
* [multi-node training](#multi-node-training)
* [mixed precision training](#mixed-precision-training)
* [SGD optimizer and learning rate scheduling](#sgd-optimizer-and-learning-rate-scheduling)
* [model oversubscription](#model-oversubscription)

**NOTE**: Multi-node training and mixed precision training can be used simultaneously.

### Multi-Node Training ###
Multi-node training makes it easy to train an embedding table of arbitrary size. In a multi-node solution, the sparse model, which is referred to as the embedding layer, is distributed across the nodes. Meanwhile, the dense model, such as DNN, is data parallel and contains a copy of the dense model in each GPU (see Fig. 2). In our implementation, HugeCTR leverages NCCL and [gossip](https://github.com/Funatiq/gossip) for high speed and scalable inter- and intra-node communication.

To run with multiple nodes, HugeCTR should be built with OpenMPI. GPUDirect support is recommended for high performance. Additionally, the configuration file and model files should be located in the Network File System and be visible to each of the processes. Here's an example of how your command should be set up when running in two nodes:
```shell
$ mpirun -N2 ./huge_ctr --train config.json
```

### Mixed Precision Training ###
Mixed precision training is supported to help improve and reduce the memory throughput footprint. In this mode, TensorCores are used to boost performance for matrix multiplication-based layers, such as `FullyConnectedLayer` and `InteractionLayer`, on Volta, Turing, and Ampere architectures. For the other layers, including embeddings, the data type is changed to FP16 so that both memory bandwidth and capacity are saved. To enable mixed precision mode, specify the mixed_precision option in the configuration file. When [`mixed_precision`](https://arxiv.org/abs/1710.03740) is set, the full FP16 pipeline will be triggered. Please note that loss scaling will be applied to avoid the arithmetic underflow (see Fig. 5). Mixed precision training can be enabled using the configuration file.

<div align=center><img width="539" height="337" src="user_guide_src/fig4_arithmetic_underflow.png"/></div>
<div align=center>Fig. 5: Arithmetic Underflow</div>

<br></br>

### SGD Optimizer and Learning Rate Scheduling ###
Learning rate scheduling allows users to configure its hyperparameters. You can set the base learning rate (`learning_rate`), number of initial steps used for warm-up (`warmup_steps`), when the learning rate decay starts (`decay_start`), and the decay period in step (`decay_steps`). Fig. 6 illustrates how these hyperparameters interact with the actual learning rate.

For example:
```json
"optimizer": {
  "type": "SGD",
  "update_type": "Local",
  "sgd_hparam": {
    "learning_rate": 24.0,
    "warmup_steps": 8000,
    "decay_start": 48000,
    "decay_steps": 24000
  }
}
```

<div align=center><img width="439" height="282" src="user_guide_src/learning_rate_scheduling.png"/></div>
<div align=center>Fig. 6: Learning Rate Scheduling</div>

<br></br>

### Model Oversubscription ###
Model oversubscription gives you the ability to load a subset of an embedding table, which exceeds the device memory capacity of single-GPU or multi-GPU machines, into the GPU(s) in a coarse grained, on-demand manner during the training stage. To use this feature, you need to split your dataset into multiple sub-datasets while extracting the unique key sets from them. This feature can only currently be used with [`Norm`](./configuration_file_setup.md#norm) and [`Raw`](./configuration_file_setup.md#raw) dataset formats. This feature will eventually support all embedding types and dataset formats. We revised our [`criteo2hugectr` tool](../tools/criteo_script/criteo2hugectr.cpp) to support the key set extraction for the Criteo dataset. For additional information, see our [Python Jupyter Notebook](../notebooks/python_interface.ipynb) to learn how to use this feature with the Criteo dataset. Please note that The Criteo dataset is a common use case, but model prefetching is not limited to only this dataset.

## Tools ##
We currently support the following tools:
* [Data Generator](#generating-synthetic-data-and-benchmarks): A configurable dummy data generator used to generate a synthetic dataset without modifying the configuration file for benchmarking and research purposes.
* [Preprocessing Script](#downloading-and-preprocessing-datasets): A set of scripts to convert the original Criteo dataset into HugeCTR using supported dataset formats such as Norm and RAW. It's used in all of our samples to prepare the data and train various recommender models.

### Generating Synthetic Data and Benchmarks
The [Norm](./configuration_file_setup.md#norm) (with Header) and [Raw](./configuration_file_setup.md#raw) (without Header) datasets can be generated with `data_generator`. For categorical features, you can configure the probability distribution to be uniform or power-law.
The default distribution is uniform.
- Using the `Norm` dataset format, run the following command: <br>
```bash
$ data_generator --config-file your_config.json --voc-size-array <vocabulary size array in csv>  --distribution <powerlaw | unified> [option: --nnz-array <nnz array in csv: all one hot>] [option: --alpha xxx or --longtail <long | medium | short>] [option:--data-folder <folder_path: ./>] [option:--files <number of files: 128>] [option:--samples <samples per file: 40960>]
$ huge_ctr --train your_config.json
```
- Using the `Raw` dataset format, run the following command: <br>
```bash
$ data_generator --config-file your_config.json --distribution <powerlaw | unified> [option: --nnz-array <nnz array in csv: all one hot>] [option: --alpha xxx or --longtail <long | medium | short>]
$ huge_ctr --train your_config.json
```

Set the following parameters:
+ `config-file`: The json configure file you used in your training. Data generator will read the configure file to get necessary data information.
+ `data_folder`: Directory where the generated dataset is stored. The default value is `./`
+ `voc-size-array`: vocabulary size per slot of your target dataset. For example, for a six slots dataset "--voc-size-array 100,23,111,45,23,2452". Note that no space between the numbers. 
+ `nnz-array`: You can use this parameter to simulate one-hot or multi-hot encodings. If you just want to use the one-hot encoding, you don't need to specify this option. If it's specified, the length of array should be the same as `voc-size-array` for norm format or `slot_size_array` in json configure file in data layer.
+ `files`: Number of data files that will be generated (optional). The default value is `128`.
+ `samples`: Number of samples per file (optional). The default value is `40960`.
+ `distribution`: Both `powerlaw` and `unified` distribution are supported.
+ `alpha`: if `powerlaw` is specified, users can specify either `alpha` or `long-tail` to configure the distribution.  
+ `long-tail`: If you want to generate data with power-law distribution for categorical features, you can use this option. You can choose from the `long`, `medium` and `short` options, which characterize the properties of the tail. The scaling exponent will be 1, 3, and 5 respectively.

Here are two examples of generating an one-hot dataset where the vocabulary size is 434428 based on the DCN config file.
```bash
$ ./data_generator --config-file dcn.json --voc-size-array 39884,39043,17289,7420,20263,3,7120,1543,39884,39043,17289,7420,20263,3,7120,1543,63,63,39884,39043,17289,7420,20263,3,7120,1543 --distribution powerlaw --alpha -1.2
$ ./data_generator --config-file dcn.json --voc-size-array 39884,39043,17289,7420,20263,3,7120,1543,39884,39043,17289,7420,20263,3,7120,1543,63,63,39884,39043,17289,7420,20263,3,7120,1543 --nnz-array 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 --distribution powerlaw --alpha -1.2
```

Here is an example of generating an one-hot dataset for DLRM config file.
```bash
$ ./data_generator --config-file dlrm_fp16_64k.json --distribution powerlaw --alpha -1.2
```

### Downloading and Preprocessing Datasets
Download the Criteo 1TB Click Logs dataset using `HugeCTR/tools/preprocess.sh` and preprocess it to train the DCN.
Then, you will find `file_list.txt`, `file_list_test.txt`, and preprocessed data files inside `criteo_data` directory. For more detailed usage, check out our [samples](../samples).

For example:
```bash
$ cd tools # assume that the downloaded dataset is here
$ bash preprocess.sh 1 criteo_data pandas 1 0
```

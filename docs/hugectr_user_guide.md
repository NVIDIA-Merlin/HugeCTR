HugeCTR User Guide
==================

HugeCTR is a GPU-accelerated framework designed to distribute training across multiple GPUs and nodes and estimate click-through rates (CTRs). HugeCTR supports model-parallel embedding tables and data-parallel neural networks and their variants such as [Wide and Deep Learning (WDL)](https://arxiv.org/abs/1606.07792), [Deep Cross Network (DCN)](https://arxiv.org/abs/1708.05123), [DeepFM](https://arxiv.org/abs/1703.04247), and [Deep Learning Recommendation Model (DLRM)](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/). HugeCTR is a component of [NVIDIA Merlin Open Beta](https://developer.nvidia.com/nvidia-merlin#getstarted). NVIDIA Merlin is used for building large-scale recommender systems, which require massive datasets to train, particularly for deep learning based solutions.

<div align=center><img src ="user_guide_src/merlin_arch.png"/></div>
<div align=center>Fig. 1: Merlin Architecture</div>

<br></br>

To prevent data loading from becoming a major bottleneck during training, HugeCTR contains a dedicated data reader that is inherently asynchronous and multi-threaded. It will read a batched set of data records in which each record consists of high-dimensional, extremely sparse, or categorical features. Each record can also include dense numerical features, which can be fed directly to the fully connected layers. An embedding layer is used to compress the input-sparse features to lower-dimensional, dense-embedding vectors. There are three GPU-accelerated embedding stages:
* table lookup
* weight reduction within each slot
* weight concatenation across the slots

To enable large embedding training, the embedding table in HugeCTR is model parallel and distributed across all GPUs in a homogeneous cluster, which consists of multiple nodes. Each GPU has its own:
* feed-forward neural network (data parallelism) to estimate CTRs
* hash table to make the data preprocessing easier and enable dynamic insertion

Embedding initialization is not required before training takes place since the input training data are hash values (64bit long long type) instead of original indices. A pair of <key,value> (random small weight) will be inserted during runtime only when a new key appears in the training data and the hash table cannot find it.

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
To obtain the detailed HugeCTR software stack (dependencies), see [Software Stack](../tools/dockerfiles/software_stack.md).

### Installing HugeCTR Using NGC Containers
All NVIDIA Merlin components are available as open source projects. However, a more convenient way to make use of these components is by using our Merlin NGC containers. Containers allow you to package your software application, libraries, dependencies, and runtime compilers in a self-contained environment. When installing HugeCTR using NGC containers, the application environment remains portable, consistent, reproducible, and agnostic to the underlying host system software configuration.

HugeCTR is included in the Merlin Docker image, which is available in the [NVIDIA container repository](https://ngc.nvidia.com/catalog/containers/nvidia:hugectr).

You can pull and launch the container by running the following command:
```shell
$ docker run --gpus=all --rm -it nvcr.io/nvidia/merlin/merlin-training:0.6  # Start interaction mode
```  

### Building Your Own HugeCTR Docker Container ###
To build the HugeCTR Docker container on your own, see [Build HugeCTR Docker Containers](../tools/dockerfiles).

### Building HugeCTR from Scratch
Before building HugeCTR from scratch, you should prepare the dependencies according to the instructions provided in the [Software Stack](../tools/dockerfiles/software_stack.md). After you've prepared the dependencies, download the HugeCTR repository and the third-party modules that it relies on by running the following commands:
```shell
$ git clone https://github.com/NVIDIA/HugeCTR.git
$ cd HugeCTR
$ git submodule update --init --recursive
```

You can build HugeCTR from scratch using one or any combination of the following options:
* **SM**: You can use this option to build HugeCTR with a specific compute capability (DSM=80) or multiple compute capabilities (DSM="70;75"). The default compute capability is 70, which uses the NVIDIA V100 GPU. For more information, see [Compute Capability](#compute-capability). 60 is not supported for inference deployments. For more information, see [Quick Start](https://github.com/triton-inference-server/hugectr_backend#quick-start).
* **CMAKE_BUILD_TYPE**: You can use this option to build HugeCTR with Debug or Release. When using Debug to build, HugeCTR will print more verbose logs and execute GPU tasks in a synchronous manner.
* **VAL_MODE**: You can use this option to build HugeCTR in validation mode, which was designed for framework validation. In this mode, loss of training will be shown as the average of eval_batches results. Only one thread and chunk will be used in the data reader. Performance will be lower when in validation mode. This option is set to OFF by default.
* **ENABLE_MULTINODES**: You can use this option to build HugeCTR with multi-nodes. This option is set to OFF by default. For more information, see [samples/dcn2nodes](../samples/dcn).
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
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70,80" -DCMAKE_BUILD_TYPE=Debug .. # Target is NVIDIA V100 / A100, Debug mode.
$ make -j
```

```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70,80" -DENABLE_INFERENCE=ON .. # Target is NVIDIA V100 / A100 and Validation mode on.
$ make -j
```

## Use Cases ##
Starding from v3.1, HugeCTR will not support the training with command line and configuration file. The Python interface will be the standard usage in model training. 

For more information regarding how to use the HugeCTR Python API and comprehend its API signature, see our [Python Interface Introduction](./python_interface.md).

## Core Features ##
In addition to single node and full precision training, HugeCTR supports a variety of features including the following:
* [Model Parallel Training](#model-parallel-training)
* [multi-node training](#multi-node-training)
* [mixed precision training](#mixed-precision-training)
* [SGD optimizer and learning rate scheduling](#sgd-optimizer-and-learning-rate-scheduling)
* [embedding training cache](#embedding-training-cache)
* [ONNX Converter](#onnx-converter)

**NOTE**: Multi-node training and mixed precision training can be used simultaneously.

### Model Parallel Training ###
Natively HugeCTR supports both model parallel training and data parallel to train very large models on GPUs. Features and categories of embeddings can be distributed across multiple GPUs and multiple nodes. For example, if you have 2 nodes with 8xA100 80GB GPUs you can train upto 1TB model fully on GPU (by embedding training cache you can train even larger models on the same nodes). 

To achieve the best performance on different embeddings, we support variant implementations of embedding layers. Each of these implementations targets different practical training cases:
* LocalizedSlotEmbeddingHash: the features in the same slot (feature field) will be stored in one GPU (that is why we call it “localized slot”), and different slots may be stored in different GPUs according to the index number of the slot. LocalizedSlotEmbedding is optimized for the case each embedding is smaller than the memory size of GPU. As local reduction per slot is used in LocalizedSlotEmbedding and no global reduce between GPUs, the overall data transaction in Embedding is much less than DistributedSlotEmbedding. **Note**: Users should gurantee that there's no overlap between different feature feature fields (no duplicated keys) in the input dataset.
* DistributedSlotEmbeddingHash: all the features (in different feature fields / slots) are distributed to different GPUs according to the index number of the feature, regardless of the index number of the slot. That means the features in the same slot may be stored in different GPUs (that is why we call it “distributed slot”). DistributedSlotEmbedding is made for the case some of the embeddings are larger than the memory size of GPU. As global reduction is required. DistributedSlotEmbedding has much more memory trasactions between GPUs. **Note**: Users should gurantee that there's no overlap between different feature feature fields (no duplicated keys) in the input dataset.
* LocalizedSlotEmbeddingOneHot: a specialized LocalizedSlotEmbedding which requires that the input of data is one hot and each feature field is indexed from zero (e.g. gender: 0,1; 1,2 is wrong)


### Multi-Node Training ###
Multi-node training makes it easy to train an embedding table of arbitrary size. In a multi-node solution, the sparse model, which is referred to as the embedding layer, is distributed across the nodes. Meanwhile, the dense model, such as DNN, is data parallel and contains a copy of the dense model in each GPU (see Fig. 2). In our implementation, HugeCTR leverages NCCL for high speed and scalable inter- and intra-node communication.

To run with multiple nodes, HugeCTR should be built with OpenMPI. GPUDirect support is recommended for high performance. Please find dcn multi-node training sample [here](../samples/dcn/dcn_2node_8gpu.py) 

### Mixed Precision Training ###
Mixed precision training is supported to help improve and reduce the memory throughput footprint. In this mode, TensorCores are used to boost performance for matrix multiplication-based layers, such as `FullyConnectedLayer` and `InteractionLayer`, on Volta, Turing, and Ampere architectures. For the other layers, including embeddings, the data type is changed to FP16 so that both memory bandwidth and capacity are saved. To enable mixed precision mode, specify the mixed_precision option in the configuration file. When [`mixed_precision`](https://arxiv.org/abs/1710.03740) is set, the full FP16 pipeline will be triggered. Loss scaling will be applied to avoid the arithmetic underflow (see Fig. 5). Mixed precision training can be enabled using the configuration file.

<div align=center><img width="539" height="337" src="user_guide_src/fig4_arithmetic_underflow.png"/></div>
<div align=center>Fig. 5: Arithmetic Underflow</div>

<br></br>

### SGD Optimizer and Learning Rate Scheduling ###
Learning rate scheduling allows users to configure its hyperparameters. You can set the base learning rate (`learning_rate`), number of initial steps used for warm-up (`warmup_steps`), when the learning rate decay starts (`decay_start`), and the decay period in step (`decay_steps`). Fig. 6 illustrates how these hyperparameters interact with the actual learning rate.

Please find more information under [Python Interface Introduction](./python_interface.md).

<div align=center><img width="439" height="282" src="user_guide_src/learning_rate_scheduling.png"/></div>
<div align=center>Fig. 6: Learning Rate Scheduling</div>

<br></br>

### Embedding Training Cache ###
Embedding Training Cache (Model oversubscription) gives you the ability to train a large model up to TeraBytes. It's implemented by loading a subset of an embedding table, which exceeds the aggregated capacity of GPU's memory, into the GPU in a coarse-grained, on-demand manner during the training stage. To use this feature, you need to split your dataset into multiple sub-datasets while extracting the unique key sets from them (see Fig. 7).<br/>This feature currently supports both single and multi-node training. It supports all embedding types and can be used with [Norm](./python_interface.md#norm) and [Raw](./python_interface.md#raw) dataset formats. We revised our [`criteo2hugectr` tool](../tools/criteo_script/criteo2hugectr.cpp) to support the key set extraction for the Criteo dataset. For additional information, see our [Python Jupyter Notebook](../notebooks/python_interface.ipynb) to learn how to use this feature with the Criteo dataset. Please note that The Criteo dataset is a common use case, but model prefetching is not limited to this dataset.

<div align=center><img width="520" height="153" src="user_guide_src/dataset_split.png"/></div>
<div align=center>Fig. 7: Preprocessing of dataset for model oversubscription</div>

### ONNX Converter ###
ONNX Converter is a python package `hugectr2onnx` that can convert HugeCTR models to ONNX format. It can improve the compatibility of HugeCTR with other deep learning frameworks given that Open Neural Network Exchange (ONNX) serves as an open-source format for AI models.

After training with HugeCTR Python APIs, you can get the files for dense model, sparse model(s) and graph configuration JSON, which are required as inputs by the method `hugectr2onnx.converter.convert`. Each HugeCTR layer will correspond to one or several ONNX operators, and the trained model weights will be loaded as initializers in the ONNX graph. Besides, users can choose to convert the sparse embedding layers or not with the flag `convert_embedding`. For more details about this feature, please refer to [ONNX Converter](../onnx_converter). There is also a notebook [hugectr2onnx_demo.ipynb](../notebooks/hugectr2onnx_demo.ipynb) that demonstrates the usage.


## Tools ##
We currently support the following tools:
* [Data Generator](#generating-synthetic-data-and-benchmarks): A configurable data generator with Python interface that can be used to generate a synthetic dataset for benchmarking and research purposes.
* [Preprocessing Script](#downloading-and-preprocessing-datasets): A set of scripts to convert the original Criteo dataset into HugeCTR using supported dataset formats such as Norm and RAW. It's used in all of our samples to prepare the data and train various recommender models.

### Generating Synthetic Data and Benchmarks
The [Norm](./python_interface.md#norm) (with Header) and [Raw](./python_interface.md#raw) (without Header) datasets can be generated with [hugectr.tools.DataGenerator](./python_interface.md#datagenerator). For categorical features, you can configure the probability distribution to be uniform or power-law within [hugectr.tools.DataGeneratorParam](./python_interface.md#datageneratorparams-class). The default distribution is power law with alpha = 1.2.

- Generate the `Norm` dataset for DCN and start training the HugeCTR model: <br>
```bash
python3 ../tools/data_generator/dcn_norm_generate_train.py
```

- Generate the `Norm` dataset for WDL and start training the HugeCTR model: <br>
```bash
python3 ../tools/data_generator/wdl_norm_generate_train.py
```

- Generate the `Raw` dataset for DLRM and start training the HugeCTR model: <br>
```bash
python3 ../tools/data_generator/dlrm_raw_generate_train.py
```

- Generate the `Parquet` dataset for DCN and start training the HugeCTR model: <br>
```bash
python3 ../tools/data_generator/dcn_parquet_generate_train.py
```

### Downloading and Preprocessing Datasets
Download the Criteo 1TB Click Logs dataset using `HugeCTR/tools/preprocess.sh` and preprocess it to train the DCN. The `file_list.txt`, `file_list_test.txt`, and preprocessed data files are available within the `criteo_data` directory. For more detailed usage, check out our [samples](../samples).

For example:
```bash
$ cd tools # assume that the downloaded dataset is here
$ bash preprocess.sh 1 criteo_data pandas 1 0
```

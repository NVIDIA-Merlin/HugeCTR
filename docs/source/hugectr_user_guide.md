# Introduction to HugeCTR

## About HugeCTR

HugeCTR is a GPU-accelerated framework designed to distribute training across multiple GPUs and nodes and estimate click-through rates (CTRs). HugeCTR supports model-parallel embedding tables and data-parallel neural networks and their variants such as [Wide and Deep Learning (WDL)](https://arxiv.org/abs/1606.07792), [Deep Cross Network (DCN)](https://arxiv.org/abs/1708.05123), [DeepFM](https://arxiv.org/abs/1703.04247), and [Deep Learning Recommendation Model (DLRM)](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/). HugeCTR is a component of [NVIDIA Merlin Open Beta](https://developer.nvidia.com/nvidia-merlin#getstarted). NVIDIA Merlin is used for building large-scale recommender systems, which require massive datasets to train, particularly for deep learning based solutions.

<br><img src="user_guide_src/merlin_arch.png"></br>
<div align=center>Fig. 1: Merlin Architecture</div>

<br></br>

To prevent data loading from becoming a major bottleneck during training, HugeCTR contains a dedicated data reader that is inherently asynchronous and multi-threaded. It will read a batched set of data records in which each record consists of high-dimensional, extremely sparse, or categorical features. Each record can also include dense numerical features, which can be fed directly to the fully connected layers. An embedding layer is used to compress the sparse input features to lower-dimensional, dense embedding vectors. There are three GPU-accelerated embedding stages:

* Table lookup
* Weight reduction within each slot
* Weight concatenation across the slots

To enable large embedding training, the embedding table in HugeCTR is model parallel and distributed across all GPUs in a homogeneous cluster, which consists of multiple nodes. Each GPU has its own:

* feed-forward neural network (data parallelism) to estimate CTRs.
* hash table to make the data preprocessing easier and enable dynamic insertion.

Embedding initialization is not required before training takes place since the input training data are hash values (64-bit signed integer type) instead of original indices. A pair of <key,value> (random small weight) will be inserted during runtime only when a new key appears in the training data and the hash table cannot find it.

<img src="user_guide_src/fig1_hugectr_arch.png" width="781px" align="center"/>

<div align=center>Fig. 2: HugeCTR Architecture</div>

<br></br>

<img src="user_guide_src/fig2_embedding_mlp.png" alt="Embedding architecture" width="500px" align="center"/>

<div align=center>Fig. 3: Embedding Architecture</div>

<br></br>

<img src="user_guide_src/fig3_embedding_mech.png" width="640px" align="center"/>

<div align=center>Fig. 4: Embedding Mechanism</div>

<br></br>

## Installing and Building HugeCTR

You can either install HugeCTR easily using the Merlin Docker image in NGC, or build HugeCTR from scratch using various build options if you're an advanced user.

### Compute Capability

We support the following compute capabilities:

| Compute Capability | GPU                  | [SM](#building-hugectr-from-scratch) |
|--------------------|----------------------|----|
| 6.0                | NVIDIA P100 (Pascal) | 60 |
| 7.0                | NVIDIA V100 (Volta)  | 70 |
| 7.5                | NVIDIA T4 (Turing)   | 75 |
| 8.0                | NVIDIA A100 (Ampere) | 80 |

### Installing HugeCTR Using NGC Containers

All NVIDIA Merlin components are available as open source projects. However, a more convenient way to utilize these components is by using our Merlin NGC containers. These containers allow you to package your software application, libraries, dependencies, and runtime compilers in a self-contained environment. When installing HugeCTR using NGC containers, the application environment remains portable, consistent, reproducible, and agnostic to the underlying host system's software configuration.

HugeCTR is included in the Merlin Docker containers that are available from the [NVIDIA container repository](https://catalog.ngc.nvidia.com/containers).
You can query the collection for containers that [match the HugeCTR label](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=scoreDESC&query=label:%22HugeCTR%22).
The following table also identifies the containers:

| Container Name             | Container Location | Functionality |
| -------------------------- | ------------------ | ------------- |
| merlin-inference           | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-inference | NVTabular, HugeCTR, and Triton Inference |
| merlin-training            | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-training | NVTabular and HugeCTR                    |
| merlin-tensorflow-training | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-training | NVTabular, TensorFlow, and HugeCTR Tensorflow Embedding plugin |

To use these Docker containers, you'll first need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker. You can use the NGC links referenced in the table above to obtain more information about how to launch and run these containers.

The following sample command pulls and starts the Merlin Training container:

```shell
# Run the container in interactive mode
$ docker run --gpus=all --rm -it --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-training:22.04
```

### Building HugeCTR from Scratch

To build HugeCTR from scratch, refer to [Build HugeCTR from source code](./hugectr_contributor_guide.md#build-hugectr-from-source).

## Tools

We currently support the following tools:
* [Data Generator](#generating-synthetic-data-and-benchmarks): A configurable data generator, which is available from the Python interface, can be used to generate a synthetic dataset for benchmarking and research purposes.
* [Preprocessing Script](#downloading-and-preprocessing-datasets): We provide a set of scripts that form a template implementation to demonstrate how complex datasets, such as the original Criteo dataset, can be converted into HugeCTR using supported dataset formats such as Norm and RAW. It's used in all of our samples to prepare the data and train various recommender models.

### Generating Synthetic Data and Benchmarks

The [Norm](/api/python_interface.md#norm) (with Header) and [Raw](/api/python_interface.md#raw) (without Header) datasets can be generated with [hugectr.tools.DataGenerator](/api/python_interface.md#datagenerator). For categorical features, you can configure the probability distribution to be uniform or power-law within [hugectr.tools.DataGeneratorParam](/api/python_interface.md#datageneratorparams-class). The default distribution is power law with alpha = 1.2.

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

Download the Criteo 1TB Click Logs dataset using `HugeCTR/tools/preprocess.sh` and preprocess it to train the DCN. The `file_list.txt`, `file_list_test.txt`, and preprocessed data files are available within the `criteo_data` directory. For more information, refer to the [samples](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/samples) directory on GitHub.

For example:
```bash
$ cd tools # assume that the downloaded dataset is here
$ bash preprocess.sh 1 criteo_data pandas 1 0
```

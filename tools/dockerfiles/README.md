# Build HugeCTR Docker Containers

As of v3.0 release, the HugeCTR container is no longer being released separately. The unified Merlin container is now available on the NVIDIA GPU Cloud (NGC) and can be used for research and production purposes. It provides support for preprocessing data using NVTabular, model training using HugeCTR, Inference using the Triton HugeCTR Backend, and the Embedding TensorFlow (TF) plugin. If you want to use any one of these use cases with the exception of preprocessing data using NVTabular, you can build a container using the dockerfile on your own.

## Build Container for Model Training

The `train.Dockerfile` supports the `devel` and `release` build modes. A container in `release` mode contains the necessary libraries and executable files so that HugeCTR and its Python interface can be used, but with no source code/dataset/script stored in it. A container in `devel` mode can be used for development purposes. After running the container, you can download the HugeCTR source code and build following the steps shown [here](../../docs/hugectr_user_guide.md#building-hugectr-from-scratch).

You can build the container by running one of these commands from the current directory:

```
# release mode
docker build -t hugectr:release -f train.Dockerfile --build-arg RELEASE=true .

# devel mode
docker build -t hugectr:devel -f train.Dockerfile .
```

## Build a Container for Inference

If you want to use the HugeCTR inference functionality based on the [Triton Inference Server HugeCTR Backend](https://github.com/triton-inference-server/hugectr_backend), build a container by running the following command from the current directory:

```
docker build -t hugectr:inference -f infer.Dockerfile --build-arg RELEASE=true .
```

## Build a Container for the Embedding TF Plugin

If you want to use the [HugeCTR embedding TF plugin](../embedding_plugin), build a container by running the following command from the current directory:

```
docker build -t hugectr:plugin-embedding -f plugin-embedding.Dockerfile .
```

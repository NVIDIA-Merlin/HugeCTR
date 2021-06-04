# Build HugeCTR Docker Containers

From the v3.0 release, we stop releasing the HugeCTR container separately. Instead, Merlin unified container is available on the NVIDIA GPU Could (NGC), which gathers the functionalities of 

1. Data preprocessing using NVTabular
2. Model training using HugeCTR
3. Inference using the Triton HugeCTR Backend
4. Embedding TensorFlow (TF) plugin

You may want to use the Merlin NGC container for both research and production purposes. But if you want to build a container from the dockerfile by yourself, please refer to the following content.

## Build Container for Model Training

The `Dockerfile.train` supports two build mode: `devel` and `release`,

* A container in `release` mode contains necessary libraries and executable files to use HugeCTR and its Python interface,  but with no source code/dataset/script stored in it. 
* A container in `devel` mode is for the development purpose. After running this container, users should download the HugeCTR source code and build following the steps shown [here](../../docs/hugectr_user_guide.md#building-hugectr-from-scratch).

Under the current directory, you can build the container by

```
# release mode
docker build -t hugectr:release -f Dockerfile.train --build-arg RELEASE=true .

# devel mode
docker build -t hugectr:devel -f Dockerfile.train .
```



## Build Container for Inference

If you want to use the HugeCTR inference functionality based on the [Triton Inference Server HugeCTR Backend](https://github.com/triton-inference-server/hugectr_backend), please build a container under the current directory by

```
docker build -t hugectr:inference -f Dockerfile.inference --build-arg RELEASE=true .
```



## Build Container for Embedding TF Plugin

If you want to use the HugeCTR embedding TF plugin described [here](../embedding_plugin), please build a container under the current directory by

```
docker build -t hugectr:plugin-embedding -f Dockerfile.plugin-embedding .
```
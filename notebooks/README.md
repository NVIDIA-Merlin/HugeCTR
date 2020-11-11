# HugeCTR Jupyter demo notebooks
This directory contains a set of Jupyter Notebook demos for HugeCTR.

## 1. Requirements

The quickest way to run a notebook here is with a docker container, which provides a self-contained, isolated and re-producible environment for various, repetitive experiments.

First, clone the repository:

```
git clone https://github.com/NVIDIA/HugeCTR
```

Next, follow the steps in the [README](../README.md#2-build-docker-image-and-hugectr) to build the NVIDIA HugeCTR container. Briefly the steps are as follows.

Inside the root directory of the HugeCTR repository, run the following command:
```
docker build -t hugectr:devel -f ./tools/dockerfiles/dev.Dockerfile .
```

* Note: If you want to try [**HugeCTR Embedding Plugin for Tensorflow** demo](embedding_plugin.ipynb), run the following command instead:
```
docker build -t hugectr:devel -f ./tools/dockerfiles/dev.tfplugin.Dockerfile .
```


Then launch the container in interactive mode (mount the root directory into the container for your convenience):

```
docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr hugectr:devel
```

Within the docker interactive bash session, build HugeCTR:
```
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Asuming the target GPU is Volta-generation GPU, such as the Tesla V100
make -j
```

Finally install and start Jupyter with

```
pip3 install --upgrade notebook
cd /hugectr
jupyter-notebook --allow-root --ip 0.0.0.0 --port 8888
```

Connect to your host machine at the port 8888 with its IP address or name in your web browser: ```http://[host machine]:8888```

Use the token available from the output of running the command above to log in, for example:

```http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b```


Within the container, the notebooks per se are located at `/hugectr/notebooks`.

## 2. Notebook List

- [movie-lens-example.ipynb](movie-lens-example.ipynb): How to train and inference with the MoveLense dataset.
- [embedding_plugin.ipynb](embedding_plugin.ipynb): How to install and use the HugeCTR embedding plugin with Tensorflow.
- [python_interface.ipynb](python_interface.ipynb): How to use the Python interface and the model oversubscribing feature.

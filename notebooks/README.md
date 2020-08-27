# HugeCTR Jupyter demo notebooks
This folder contains demo notebooks for HugeCTR.

## 1. Requirements

The most convenient way to run these notebooks is via a docker container, which provides a self-contained, isolated and re-producible environment for all experiments.

First, clone the repository:

```
git clone https://github.com/NVIDIA/HugeCTR
```

Next, follow the step in the [README](../README.md) to build the NVIDIA HugeCTR container. Briefly, the steps are as follows.

From repo root:

```
docker build -t hugectr:devel -f ./tools/dockerfiles/dev.Dockerfile .
```

Then launch the container in interaction mode (mount the home directory of repo into container for easy development):

```
docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr hugectr:devel bash
```

Within the docker interactive bash session, build HugeCTR:
```
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Asuming the target GPU is Volta-generation GPU, such as the Tesla V100
make -j
```

Then finally, install and start Jupyter with

```
pip3 install jupyter
cd /hugectr
jupyter notebook --allow-root --ip 0.0.0.0 --port 8888
```

Navigate a web browser to the IP address or hostname of the host machine
at port 8888: ```http://[host machine]:8888```

Use the token listed in the output from running the jupyter command to log
in, for example:

```http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b```


Within the container, the notebooks themselves are located at `/hugectr/notebooks`.

## 2. Notebook list

- [movie-lens-example.ipynb](movie-lens-example.ipynb): Training and inference demo on the movie lens dataset.

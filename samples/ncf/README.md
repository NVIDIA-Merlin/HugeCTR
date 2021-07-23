# NCF CTR SAMPLE #
The purpose of this sample is to demonstrate how to build and train an [NCF model](https://arxiv.org/abs/1708.05031) with HugeCTR.

## Set Up the HugeCTR Docker Environment ##
You can set up the HugeCTR Docker environment by doing one of the following:
- [Pull the NGC Docker](#pull-the-ngc-docker)
- [Build the HugeCTR Docker Container on Your Own](#build-the-hugectr-docker-container-on-your-own)

### Pull the NGC Docker ###
HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to pull the pre-built Docker image, which is available on the NVIDIA GPU Cloud (NGC). This method provides a self-contained, isolated, and reproducible environment for repetitive experiments.

1. Pull the HugeCTR NGC Docker by running the following command:
   ```bash
   $ docker pull nvcr.io/nvidia/merlin/merlin-training:0.6
   ```
2. Launch the container in interactive mode with the HugeCTR root directory mounted into the container by running the following command:
   ```bash
   $ docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-training:0.6
   ```

### Build the HugeCTR Docker Container on Your Own ###
Please refer to [Build HugeCTR Docker Containers](../../tools/dockerfiles#build-container-for-model-training) to build on your own and set up the Docker container. Make sure that HugeCTR is built and installed to the system path `/usr/local/hugectr` within the Docker container. Launch the container in interactive mode in the same manner as above, and then set the `PYTHONPATH` environment variable inside the Docker container using the following command:
```shell
$ export PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH
```
## Download and Preprocess the MovieLens data ##
The [Movielens dataset](https://grouplens.org/datasets/movielens/) provided by GroupLens Research is used in this example, but the provided prerocessing scripts can be edited to format other user-item interaction data for use with this example. Scripts are provided to download and preprocess both the 1M and 20M MovieLens datasets, and minor editing of the model definition is needed to use this sample code on other datasets. The default 20M MovieLens dataset used in this example contains 20 million interactions between 138493 users and 26744 items.  After preprocessing, each interaction is simply a userId and itemId.

The provided script `get_ml20_dataset.sh` downloads and extracts the MovieLens 20M dataset. The dataset can then be prepared for training of the NCF model by running the provided python script `preprocess-20m.py`.  To run the preprocessing python script in the docker container, first install pytorch with the command:

```shell
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Running `python preprocess-20m.py` generates a series of negative interactions to add to the dataset and writes the combined result out to a series of files that can be used by HugeCTR to train the NCF model.

## Train and validate the NCF model ##
With the above steps, the preprocessed data is saved locally in the `data/` directory. To train the NCF model using the data, simply run:
``` shell
$ python ncf.py
```
By default, this will run 10 epochs of the dataset and provide Cumulative Hit Rate (HitRate) accuracy results after each.  The HitRate provided by HugeCTR is computed on testing data as the fraction of predictions that are over a threshold (0.8) and correspond to a real interaction (i.e., label is 1). 

If you are using the `Movielens 1M` dataset instead, simply run `get_ml1_data.sh` and `python preprocess-1m.py` to prepare the dataset.  Then edit `ncf.py` to use `ml-1m` directories, and change  parameters to the values that are commented out in `ncf.py` (such as `workspace_size_per_gpu_in_mb`).  Note that in general, `workspace_size_per_gpu_in_mb` should be approximately the sum of users and items in the dataset.


## Performance Evaluation ##
Example code for training the NCF model is also available with [Tensorflow](https://ngc.nvidia.com/catalog/resources/nvidia:ncf_for_tensorflow) and [PyTorch](https://ngc.nvidia.com/catalog/resources/nvidia:ncf_for_pytorch).  Below is a table with the average training time and computed hit rate after 10 epochs using different NVIDIA GPU hardware using HugeCTR, Tensorflow, and PyTorch.

| GPU(s) | HitRate | 1x V100 | 8x V100 | 1x A100 |
| ------ | ------ | ------ | ------ | ------ |
| HugeCTR | 0.951* | 44.64s | 17.04s | 32.38s |
| TensorFlow | 0.959 | 62.99s | 16.03s | 49.63s |
| Pytorch | 0.953 | 94.6s | 17.14s | 49.13s |

\* NCF in HugeCTR computes the cumulative hit rate, while TensorFlow and Pytorch use top10 hit rate.  These metrics differ, so the accuracy of HugeCTR may differ from TensorFlow and Pytorch.

## Variatons of NCF ##
The [NCF model](https://arxiv.org/abs/1708.05031) is described along with 2 additional models designed for this type of data: GMF and NeuMF.  This sample directory also contains the HugeCTR model definitions for these models.  To train the GMF or NeuMF model, simply run `python gmf.py` or `python neumf.py`, respectively.  However, we find that, on this MovieLens 20M dataset, the standard NCF model (`ncf.py`) provides the best Cumulative HitRate using the fewest number of epochs.

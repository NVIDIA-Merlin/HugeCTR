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
   $ docker pull nvcr.io/nvidia/merlin/merlin-hugectr:23.12
   ```
2. Launch the container in interactive mode with the HugeCTR root directory mounted into the container by running the following command:
   ```bash
   $ docker run --gpus=all --rm -it --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-hugectr:23.12
   ```

### Build the HugeCTR Docker Container on Your Own ###
Please refer to [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development) to build on your own and set up the Docker container. Make sure that HugeCTR is built and installed to the system path `/usr/local/hugectr` within the Docker container. Launch the container in interactive mode in the same manner as above, and then set the `PYTHONPATH` environment variable inside the Docker container using the following command:
```shell
$ export PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH
```
## Download and Preprocess the MovieLens data ##
Please refer to [NVTabular](https://nvidia-merlin.github.io/NVTabular/v0.5.0/examples/getting-started-movielens/01-Download-Convert.html#) for data downloading and preprocessing.

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

## Variatons of NCF ##
The [NCF model](https://arxiv.org/abs/1708.05031) is described along with two additional models that are designed for this type of data: GMF and NeuMF.  This sample directory also contains the HugeCTR model definitions for these models.  To train the GMF or NeuMF model, run `python gmf.py` or `python neumf.py`, respectively.  However, we find that, on this MovieLens 20M dataset, the standard NCF model (`ncf.py`) provides the best Cumulative HitRate using the fewest number of epochs.

# DLRM WITH FTRL OPTIMIZER SAMPLE #
The purpose of this sample is to demonstrate how to build and train a [DLRM model](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/) with HugeCTR FTRL optimizer and embedding collection.

## Table of Contents
* [Set Up the HugeCTR Docker Environmen](#set-up-the-hugectr-docker-environment)
* [Kaggle DLRM](#kaggle-dlrm)

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

## Kaggle DLRM
Ensure that you've met the following requirements:
- DGX A100 or DGX2 (32GB V100)

### Preprocess the Kaggle Display Advertising Dataset ##
The Kaggle Display Advertising dataset is provided by CriteoLabs. For more information, see https://ailab.criteo.com/ressources/. The original training set contains 45,840,617 examples. Each sample consists of a label (0 if the ad wasn't clicked and 1 if the ad was clicked) and 39 features (13 integer features and 26 categorical features). The dataset is also missing numerous values across the feature columns, which should be preprocessed accordingly. The original test set doesn't contain labels, so it's not used.

1. Go ([here](https://ailab.criteo.com/ressources/)) and download the Kaggle Display Advertising Dataset into the `"${project_home}/samples/dlrm/"` folder.
   As an alternative, you can run the following commands: 
   ```bash
   # download and preprocess
   $ cd ./samples/dlrm/
   $ tar zxvf dac.tar.gz
   ```
   The `dlrm_raw` tool converts the original data to HugeCTR's raw format and fills the missing values. Only `train.txt` will be used to generate raw data.

2. Convert the dataset to the HugeCTR raw data format by running the following command:
   ```bash
   # Usage: dlrm_raw input_dir out_dir
   $ dlrm_raw ./ ./ 
   ```
   This operation will generate `train.bin(5.5GB)` and `test.bin(700MB)`.

### Run the Kaggle Display Advertising Dataset ##

Way to enable Ftrl optimizer
```python
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Ftrl,
                                    update_type = hugectr.Update_t.Global,
                                    beta = 0.9,
                                    lambda1 = 0.1,
                                    lambda2 = 0.1)
```

Train with HugeCTR by running the following command:
   ```bash
   $ python3 dlrm_train_ftrl.py --shard_plan uniform --use_dynamic_hash_table --optimizer ftrl
   ```

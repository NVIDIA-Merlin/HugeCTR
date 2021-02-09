# DLRM CTR SAMPLE #
A sample of building and training DLRM model with HugeCTR [(link)](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/).

## Setup the HugeCTR Docker Environment ##
The quickest way to run the sample here is with a docker container, which provides a self-contained, isolated, and reproducible environment for repetitive experiments. HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to use the pre-built Docker image available from the NVIDIA GPU Cloud (NGC). If you want to build the HugeCTR docker image on your own, please refer to [Use Docker Container](../docs/mainpage.md#use-docker-container).

You can choose either to pull the NGC docker or to build on your own.

#### Pull the NGC Docker ####
Pull the HugeCTR NGC docker using this command:
```bash
$ docker pull nvcr.io/nvidia/hugectr:v3.0
```
Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command:
```bash
$ docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/hugectr:v3.0
```

#### Build on Your Own ####
Please refer to [Use Docker Container](../docs/mainpage.md#use-docker-container) to build on your own and set up the docker container. Please make sure that HugeCTR is built and installed to the system path `/usr/local/hugectr` within the docker container. Please launch the container in interactive mode in the same manner as above.

## Dataset and Preprocess for Terabyte Click Logs ##
[(Terabyte Click Logs)](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) provided by CriteoLabs is used in this sample. The up limit of row count of each embedding table is limited to 40 million.
We process the data in the same way as dlrm [(repo)](https://github.com/facebookresearch/dlrm#benchmarking). Each sample has 40 32bits integers, where the first integer is label,
the next 13 integers are dense features and the following 26 integers are category features.

## Requirements ##
* Requirements on [README](../../README.md#Requirements) 
* DGX A100 or DGX2 (32GB V100)


## Run ##
1. Download TeraBytes datasets from [(Terabyte Click Logs)](https://labs.criteo.com/2013/12/download-terabyte-click-logs/). Unzip them and name as `day_0`, `day_1`, ..., `day_23`.

2. Preprocess Datasets. This operation will generate `train.bin(671.2GB)` and `test.bin(14.3GB)`.
```bash
# Usage: dlrm_raw input_dir output_dir --train {days for training} --test {days for testing}
$ dlrm_raw ./ ./ \
--train 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
--test 23
```

3. Run either of the four json configure files in this directory: e.g.
```shell
$ huge_ctr --train ./terabyte_fp16_64k.json
```

* **Note**: In v2.2.1, there is a CUDA Graph related issue in running this sample on DGX2. To run it on DGX2, specify `"cuda_graph": false` in `solver` section of your JSON config.
For more detailed information on this error, check [Known Issues](docs/hugectr_user_guide.md#known-issues) section.
* **Note**: `cache_eval_data` is only supported on DGX A100. If your machine is DGX2, disable it. 

## Dataset and preprocess for Kaggle Criteo Logs ##
The data is provided by CriteoLabs (https://ailab.criteo.com/ressources/) as `Kaggle Display Advertising dataset`.
The original training set contains 45,840,617 examples.
Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
The dataset also has the significant amounts of missing values across the feature columns, which should be preprocessed acordingly.
The original test set doesn't contain labels, so it's not used.

## Requirements ##
+ Python >= 3.6.9
+ libcudf >= 0.15
+ librmm >= 0.15


## Run ##
1. Download the dataset and preprocess <br>
Go to ([link](https://ailab.criteo.com/ressources/)) and download kaggle-display dataset into the folder `"${project_home}/samples/dlrm/"`. The tool `dlrm_raw` converts original datas to HugeCTR's raw format and fills the missing values. Only `train.txt` will be used to generate raw datas.
```bash
# download and preprocess
$ cd ./samples/dlrm/
$ tar zxvf dac.tar.gz
```

2. Convert the dataset to HugeCTR Raw format
```bash
# Usage: dlrm_raw input_dir out_dir
$ dlrm_raw ./ ./ 
```

3. Train with HugeCTR
```bash
$ huge_ctr --train ./kaggle_fp32.json
```

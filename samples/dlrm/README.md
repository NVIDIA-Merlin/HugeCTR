# DLRM CTR SAMPLE #
The purpose of this sample is to demonstrate how to build and train a [DLRM model](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/) with HugeCTR.

## Set Up the HugeCTR Docker Environment ##
You can set up the HugeCTR Docker environment by doing one of the following:
- [Pull the NGC Docker](#pull-the-ngc-docker)
- [Build the HugeCTR Docker Container on Your Own](#build-the-hugectr-docker-container-on-your-own)

### Pull the NGC Docker ###
HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to pull the pre-built Docker image, which is available on the NVIDIA GPU Cloud (NGC). This method provides a self-contained, isolated, and reproducible environment for repetitive experiments.

1. Pull the HugeCTR NGC Docker by running the following command:
   ```bash
   $ docker pull nvcr.io/nvidia/merlin/merlin-training:0.5
   ```
2. Launch the container in interactive mode with the HugeCTR root directory mounted into the container by running the following command:
   ```bash
   $ docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-training:0.5
   ```
3. Activate the merlin conda environment by running the following command:  
   ```shell.
   source activate merlin
   ```  

### Build the HugeCTR Docker Container on Your Own ###
Please refer to [Build HugeCTR Docker Containers](../../tools/dockerfiles#build-container-for-model-training) to build on your own and set up the Docker container. Make sure that HugeCTR is built and installed to the system path `/usr/local/hugectr` within the Docker container. Launch the container in interactive mode in the same manner as above.

## Preprocess the Terabyte Click Logs ##
The [Terabyte Click Logs](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) provided by CriteoLabs is used in this sample. The row count of each embedding table is limited to 40 million. The data is processed the same way as dlrm. For more information, see [Benchmarking](https://github.com/facebookresearch/dlrm#benchmarking). Each sample has 40 32bits integers in which the first integer is label. The next 13 integers are dense features and the following 26 integers are category features.

## Run the Terabyte Click Logs ##
Ensure that you've met the following requirements:
- See the requirements noted [here](../../README.md#Requirements) 
- DGX A100 or DGX2 (32GB V100)

1. Download the terabyte datasets from the [Terabyte Click Logs](https://labs.criteo.com/2013/12/download-terabyte-click-logs/). 

2. Unzip the datasets and name them in the following manner: `day_0`, `day_1`, ..., `day_23`.

2. Preprocess the datasets using the following command:
   ```bash
   # Usage: dlrm_raw input_dir output_dir --train {days for training} --test {days for testing}
   $ dlrm_raw ./ ./ \
   --train 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
   --test 23
   ```
   This operation will generate `train.bin(671.2GB)` and `test.bin(14.3GB)`.

3. Run any of the four JSON configuration files in this directory using the following command:
   ```shell
   $ huge_ctr --train ./terabyte_fp16_64k.json
   ```

**IMPORTANT NOTES**: 
- In v2.2.1, there is a CUDA Graph error that occurs when running this sample on DGX2. To run it on DGX2, specify `"cuda_graph": false` in the `solver` section of your JSON configuration file. For detailed information about this error, see [Known Issues](docs/hugectr_user_guide.md#known-issues).
- `cache_eval_data` is only supported on DGX A100. If you're running DGX2, disable it. 

## Preprocess the Kaggle Display Advertising Dataset ##
The Kaggle Display Advertising dataset is provided by CriteoLabs. For more information, see https://ailab.criteo.com/ressources/. The original training set contains 45,840,617 examples. Each sample consists of a label (0 if the ad wasn't clicked and 1 if the ad was clicked) and 39 features (13 integer features and 26 categorical features). The dataset is also missing numerous values across the feature columns, which should be preprocessed accordingly. The original test set doesn't contain labels, so it's not used.

Ensure that you've met the following requirements:
- Python version 3.6.9
- libcudf version 0.15
- librmm version 0.15

## Run the Kaggle Display Advertising Dataset ##
1. Go ([here](https://ailab.criteo.com/ressources/)) and download the Kaggle Display Advertising Dataset into the `"${project_home}/samples/
   dlrm/"` folder.
   As an alternative, you can run the following commands: 
   ```bash
   # download and preprocess
   $ cd ./samples/dlrm/
   $ tar zxvf dac.tar.gz
   ```
   The `dlrm_raw` tool converts the original data to HugeCTR's raw format and fills the missing values. Only `train.txt` will be used to generate raw 
   data.
2. Convert the dataset to the HugeCTR raw data format by running the following command:
   ```bash
   # Usage: dlrm_raw input_dir out_dir
   $ dlrm_raw ./ ./ 
   ```
3. Train with HugeCTR by running the following command:
   ```bash
   $ huge_ctr --train ./kaggle_fp32.json
   ```

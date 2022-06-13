# RESEARCH SAMPLE #
The purpose of this sample is to demonstrate how to build and train a sample for the sake of research purpose.

## Table of Contents
* [Set Up the HugeCTR Docker Environment](#set-up-the-hugectr-docker-environment)
* [Performance Research](#performance-research)

## Set Up the HugeCTR Docker Environment ##
You can set up the HugeCTR Docker environment by doing one of the following:
- [Pull the NGC Docker](#pull-the-ngc-docker)
- [Build the HugeCTR Docker Container on Your Own](#build-the-hugectr-docker-container-on-your-own)

### Pull the NGC Docker ###
HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to pull the pre-built Docker image, which is available on the NVIDIA GPU Cloud (NGC). This method provides a self-contained, isolated, and reproducible environment for repetitive experiments.

1. Pull the HugeCTR NGC Docker by running the following command:
   ```bash
   $ docker pull nvcr.io/nvidia/merlin/merlin-training:22.05
   ```
2. Launch the container in interactive mode with the HugeCTR root directory mounted into the container by running the following command:
   ```bash
   $ docker run --gpus=all --rm -it --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-training:22.05
   ```

### Build the HugeCTR Docker Container on Your Own ###
Please refer to [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development) to build on your own and set up the Docker container. Make sure that HugeCTR is built and installed to the system path `/usr/local/hugectr` within the Docker container. Launch the container in interactive mode in the same manner as above, and then set the `PYTHONPATH` environment variable inside the Docker container using the following command:
```shell
$ export PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH
```

## Performance Research
We will show that multiple performance optmizations can be applied to different network structures.
|                               | Mode           | Network structure highlight                   | async_mlp_wgrad | gen_loss_summary | overlap_lr | overlap_init_wgrad | overlap_ar_a2a | use_cuda_graph | use_holistic_cuda_graph | use_overlapped_pipeline | grouped_all_reduce | all_reduce_algo | data_reader_type | embedding_type  | 
| ---                           | ---            | ---                                          | ---             | ---              | ---        | ---                | ---            | ---            | ---                     | ---                     | ---                | ---             | ---              | ---             | 
| [model_1.py](./model_1.py)    | Singe-node     | MultiCross layer after interaction layer     | OFF             | OFF              | ON         | ON                 | ON             | OFF            | ON                      | ON                      | OFF                | OneShot         | RawAsync         | HybridEmbedding |
| [model_2.py](./model_2.py)    | Singe-node     | Dropout layers in the MLP                    | OFF             | OFF              | ON         | ON                 | ON             | OFF            | ON                      | ON                      | OFF                | OneShot         | RawAsync         | HybridEmbedding |
| [model_3.py](./model_3.py)    | Multi-node     | Dropout layers in the MLP                    | OFF             | OFF              | ON         | ON                 | ON             | OFF            | ON                      | ON                      | ON                 | OneShot         | RawAsync         | HybridEmbedding |
| [model_4.py](./model_4.py)    | Single-node    | Smaller embedding vector size                | OFF             | OFF              | ON         | ON                 | ON             | OFF            | ON                      | ON                      | OFF                | OneShot         | RawAsync         | HybridEmbedding |


Ensure that you've met the following requirements:
- DGX A100

### Preprocess the Kaggle Display Advertising Dataset ##
The Kaggle Display Advertising dataset is provided by CriteoLabs. For more information, see https://ailab.criteo.com/ressources/. The original training set contains 45,840,617 examples. Each sample consists of a label (0 if the ad wasn't clicked and 1 if the ad was clicked) and 39 features (13 integer features and 26 categorical features). The dataset is also missing numerous values across the feature columns, which should be preprocessed accordingly. The original test set doesn't contain labels, so it's not used.

1. Go ([here](https://ailab.criteo.com/ressources/)) and download the Kaggle Display Advertising Dataset into the `"${project_home}/samples/research/"` folder.
   As an alternative, you can run the following commands: 
   ```bash
   # download and preprocess
   $ cd ./samples/research/
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
Train with HugeCTR by running the following command (replace x with the actual number):
   ```bash
   $ python3 model_x.py
   ```

**IMPORTANT NOTES**: 
- To run the single-node DGX-A100 training scripts on a cluster platform, you need to enter the container properly and use the command `mpirun -np 1 --allow-run-as-root python3 model_x.py` within the container.
- To run the multi-node DGX-A100 training scripts on a cluster platform, you need to submit the job on the login node properly.

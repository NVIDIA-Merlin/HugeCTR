# DLRM CTR SAMPLE #

> **Deprecation Warning**: DLRM samples are based on the [one-hot RawAsync DataReader](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#raw) and HybridEmbedding, both of which were deprecated. Please check out the [multi-hot RawAsync DataReader]((https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#raw)) and [embedding collection](https://nvidia-merlin.github.io/HugeCTR/main/api/hugectr_layer_book.html#embedding-collection) for alternatives.

The purpose of this sample is to demonstrate how to build and train a [DLRM DCNv2 model](https://arxiv.org/abs/2008.13535) with HugeCTR.

## Table of Contents
* [Set Up the HugeCTR Docker Environment](#set-up-the-hugectr-docker-environment)
* [MLPerf DLRM](#mlperf-dlrm)

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

## MLPerf DLRM
Ensure that you've met the following requirements:
- MLPerf v3.1: DGX H100 1 node, 8 nodes or 16 nodes
- Install requirements: pip install -r requirements.txt

### Dataset downloading and preprocessing ##
Input preprocessing steps below are based on the instructions from the official reference implementation repository, see [Running the MLPerf DLRM v2 benchmark](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm#running-the-mlperf-dlrm-v2-benchmark). Besides, there is a final step to convert the reference implementation dataset to the raw format in order to make it consumable by HugeCTR training script. For completeness, all the steps are detailed below.

This process can take up to several days and needs 7 TB of fast storage space. The preprocessing steps do not require a GPU machine.

**1.1** Download the dataset from https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/.

**1.2** Clone the reference implementation repository.

```
git clone https://github.com/mlcommons/training.git
cd training/recommendation_v2/torchrec_dlrm
```

**1.3** Build and run the reference docker image.
```
docker build -t dlrmv2_reference .
docker run -it --rm --network=host --ipc=host -v /data:/data dlrmv2_reference
```

**1.4** Run preprocessing steps to get data in NumPy format.

```
./scripts/process_Criteo_1TB_Click_Logs_dataset.sh \
    /data/criteo_1tb/raw_input_dataset_dir \
    /data/criteo_1tb/temp_intermediate_files_dir \
    /data/criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir
```
As a result, files named: `day_*_labels.npy`, `day_*_dense.npy` and `day_0_sparse.npy` will be created (3 per each of 24 days in the original input dataset, 72 files in total). Once completed, the output data can be verified with md5sums provided in [md5sums_preprocessed_criteo_click_logs_dataset.txt](https://github.com/mlcommons/training/blob/master/recommendation_v2/torchrec_dlrm/md5sums_preprocessed_criteo_click_logs_dataset.txt) file.

**1.5** Create a synthetic multi-hot Criteo dataset.

This step produces multi-hot dataset from the original (one-hot) dataset.

```
python scripts/materialize_synthetic_multihot_dataset.py \
    --in_memory_binary_criteo_path /data/criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir \
    --output_path /data/criteo_1tb_sparse_multi_hot \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
    --multi_hot_distribution_type uniform
```

As a result, `day_*_sparse_multi_hot.npz` files will be created (24 files in total). Once done, the output data can be validated with md5sums provided in [md5sums_MLPerf_v2_synthetic_multi_hot_sparse_dataset.txt](https://github.com/mlcommons/training/blob/master/recommendation_v2/torchrec_dlrm/md5sums_MLPerf_v2_synthetic_multi_hot_sparse_dataset.txt) file.

**1.6** Convert NumPy dataset to raw format.

Because HugeCTR uses, among others, [raw format](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#raw) for input data, we need to convert NumPy files created in the preceding steps to this format. To this end, use `preprocessing/convert_to_raw.py` script that comes with the container created in section [Build the container and push to a docker registry](#build-the-container-and-push-to-a-docker-registry) below.

```
docker run -it --rm --network=host --ipc=host -v /data:/data nvcr.io/nvidia/merlin/merlin-hugectr:23.12
```
In that container, run:
```
python preprocessing/convert_to_raw.py \
   --input_dir_labels_and_dense /data/criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir \
   --input_dir_sparse_multihot /data/criteo_1tb_sparse_multi_hot \
   --output_dir /data/criteo_1tb_multihot_raw \
   --stages train val
```
As a result, `train_data.bin` and `val_data.bin` will be created. Once done, the output files can be verified with the md5sums provided in `preprocessing/md5sums_raw_dataset.txt` file.

### Specify the preprocessed data paths in the training script.

You may need to manually change the location of the datasets in the `train.py` file.
The `source` parameter should specify the absolute path to the `train_data.bin` file and the `eval_source`
parameter should point to the `val_data.bin` file from `/data/criteo_1tb_multihot_raw` folder obtained in the previous step.

However, for launching with nvidia-docker, you just need to make sure to set `DATADIR` as the path to the directory containing those two files.

### Steps to launch training on a single node

#### NVIDIA DGX H100 (single-node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX H100
single-node submission are in the `config_DGXH100_1x8x6912.sh` script and in the `train.py` config file.

To launch the training on a single node with a Slurm cluster run:
```
source config_DGXH100_1x8x6912.sh
CONT=<docker/registry>/mlperf-nvidia:recommendation_hugectr LOGDIR=<path/to/output/dir> sbatch -N 1 run.sub
```

Note that this benchmark has high I/O bandwidth requirements. To achieve optimal performance in the case of single-node training job at least 13.4 GB/s and 41.4 GB/s read bandwidth is required during training and evaluation stage, respectively.

#### Alternative launch with docker

When generating results for the official v3.0 submission with one node, the
benchmark was launched onto a cluster managed by a Slurm scheduler. The
instructions in [NVIDIA DGX H100 (single node)](#nvidia-dgx-h100-single-node) explain
how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using `nvidia-docker`. Note that performance or functionality may
vary from the tested Slurm instructions.

```
source config_DGXH100_1x8x6912.sh
CONT=<docker/registry>mlperf-nvidia:recommendation_hugectr DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```

### Steps to launch training on multiple nodes

#### NVIDIA DGX H100 (multi-node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX H100
multi-node submission are in the `config_DGXH100_8x8x2112.sh` or `config_DGXH100_16x8x1056.sh` scripts
and in the `train.py` config file.

To launch the training for a selected config with a Slurm cluster run:
```
source config_DGXH100_8x8x2112.sh
CONT=<docker/registry>/mlperf-nvidia:recommendation_hugectr LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES run.sub
```

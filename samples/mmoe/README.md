# MMoE SAMPLE #
The purpose of this sample is to demonstrate how to build and train a [Multi-gate Mixture of Experts (MMoE) model](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) with HugeCTR.  The sample uses the real-world dataset that was used in the [paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) that introduced the MMoE model.  The MMoE model is a multi-task model that aims to achieve high accuracy for multiple different tasks by using a mixture-of-experts approach along with separate gating layers for each task.  Below is a diagram of the model structure as implemented in HugeCTR.

## Set Up the HugeCTR Docker Environment ##
You can set up the HugeCTR Docker environment by doing one of the following:
- [Pull the NGC Docker](#pull-the-ngc-docker)
- [Build the HugeCTR Docker Container on Your Own](#build-the-hugectr-docker-container-on-your-own)

### Pull the NGC Docker ###
HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to pull the pre-built Docker image, which is available on the NVIDIA GPU Cloud (NGC). This method provides a self-contained, isolated, and reproducible environment for repetitive experiments.

1. Pull the HugeCTR NGC Docker by running the following command:
   ```bash
   $ docker pull nvcr.io/nvidia/merlin/merlin-hugectr:23.03
   ```
2. Launch the container in interactive mode with the HugeCTR root directory mounted into the container by running the following command:
   ```bash
   $ docker run --gpus=all --rm -it --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-hugectr:23.03
   ```

### Build the HugeCTR Docker Container on Your Own ###
Please refer to [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development) to build on your own and set up the Docker container. Make sure that HugeCTR is built and installed to the system path `/usr/local/hugectr` within the Docker container. Launch the container in interactive mode in the same manner as above, and then set the `PYTHONPATH` environment variable inside the Docker container using the following command:
```shell
$ export PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH
```
## Preparing your dataset for HugeCTR ##
The dataset used in this sample is the Census-UCI dataset used by the [MMoE paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007).  We provide scripts to download and preprocess the dataset so that you can try both the `MMoE` and `Shared Bottom` models on this dataset without additional work.  These scripts convert the dataset and provide labels for the two tasks descussed in the MMoE paper.  One task is to determine if the row corresponds to a person that meets an income threshold, and the other task is based on marital status.  The MMoE and Shared-bottom models aim to accurate solve both tasks with a single model. 

To download and preprocess the UCI Census dataset, simply run:
``` shell
$ ./get_dataset.sh
$ python preprocess-census.py
```
This downloads and preprocesses the dataset into Parquet format for use by HugeCTR for training.  It stores these files in the `./data` subdirectory.

## Train and validate the MMoE model ##
Once you have preprocessed the dataset and saved it in Parquet format, you can train either the `MMoE` or `Shared-bottom` model using the training script.  To train the MMoE model, simply run:
``` shell
$ python mmoe_parquet.py
```
This trains the model using roughly 200,000 samples, each with 2 labels.  AUC values are reported for each task, along with a compound AUC for both tasks.  As expected, the MMoE model reaches much higher AUC than the shared-bottom model.


### Known issues ###
- Export predictions is not currently supported for multi-task models.  This will be solved in an upcoming release.
- Accuracy metrics other than AUC are not currently supported for multi-task models.

# Sparse Operation Kit Notebooks #
This directory contains a set of Jupyter notebooks that demonstrate how to use SOK in TensorFlow.

## Quickstart

The simplest way to run a one of our notebooks is with a Docker container.
A container provides a self-contained, isolated, and reproducible environment for repetitive experiments.
Docker images are available from the NVIDIA GPU Cloud (NGC).
If you prefer to build the HugeCTR Docker image on your own, refer to [Set Up the Development Environment With Merlin Containers For SOK](https://github.com/NVIDIA-Merlin/HugeCTR/tree/main/sparse_operation_kit#obtaining-sok-and-hugectr-via-docker).

### Pull the NGC Docker

Pull the container using the following command:

```shell
docker run nvcr.io/nvidia/merlin/merlin-tensorflow:nightly
```

### Clone the HugeCTR Repository

Use the following command to clone the HugeCTR repository:

```shell
git clone https://github.com/NVIDIA/HugeCTR
```

### Start the Jupyter Notebook
1. Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command:

   ```shell
   docker run --runtime=nvidia --rm -it --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr -p 8080:8080 nvcr.io/nvidia/merlin/merlin-tensorflow:23.11
   ```

2. Start Jupyter using these commands: 

   ```shell
   cd /hugectr/hps_tf/notebooks
   jupyter-notebook --allow-root --ip 0.0.0.0 --port 8080 --NotebookApp.token='hugectr'
   ```

3. Connect to your host machine using the 8080 port by accessing its IP address or name from your web browser: `http://[host machine]:8080`

   Use the token available from the output by running the command above to log in. For example:

   `http://[host machine]:8080/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`


## Notebook List

Here's a list of notebooks that you can run:
- [sok_train_dlrm_demo.ipynb](sok_train_dlrm_demo.ipynb): Demonstrates how to train DLRM model with TF and SOK.

- [hps_inference_dlrm_with_sok_weight_demo.ipynb](hps_inference_dlrm_with_sok_weight_demo.ipynb): Demonstrateshow to use hps-tf to read SOK's sparse weight and perform inference. Before running, you must first complete the training in [sok_train_dlrm_demo.ipynb](sok_train_dlrm_demo.ipynb).

- [sok_incremental_dump_demo.ipynb](sok_incremental_dump_demo.ipynb): Demonstrates how to use SOK for incremental dump and verifies its correctness.

- [sok_dump_load_demo.ipynb](sok_dump_load_demo.ipynb): Demonstrates  how to use SOK for dump/load and verifies the correctness of the dump/load operations.

## System Specifications
The specifications of the system on which each notebook can run successfully are summarized in the table. The notebooks are verified on the system below but it does not mean the minimum requirements.

| Notebook                                                                                   | CPU                                                          | GPU                              | #GPUs | Author         |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------- | ----- | -------------- |
| [sok_train_dlrm_demo.ipynb](sok_train_dlrm_demo.ipynb)                   | AMD EPYC 7742 64-Core Processor<br />2 TB Memory | Ampere A100-SXM4-80GB<br />80 GB Memory | 2     | Hui Kang    |
| [hps_inference_dlrm_with_sok_weight_demo.ipynb](hps_inference_dlrm_with_sok_weight_demo.ipynb)                   | AMD EPYC 7742 64-Core Processor<br />2 TB Memory | Ampere A100-SXM4-80GB<br />80 GB Memory | 1     | Hui Kang    |
| [sok_incremental_dump_demo.ipynb](sok_incremental_dump_demo.ipynb)                         | AMD EPYC 7742 64-Core Processor<br />2 TB Memory | Ampere A100-SXM4-80GB<br />80 GB Memory | 2     | Hui Kang    |
| [sok_dump_load_demo.ipynb](sok_dump_load_demo.ipynb)                                       | AMD EPYC 7742 64-Core Processor<br />2 TB Memory | Ampere A100-SXM4-80GB<br />80 GB Memory | 2     | Hui Kang    |


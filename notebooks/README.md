# HugeCTR Example Notebooks

This directory contains a set of Jupyter notebook that demonstrate how to use HugeCTR.
 
The simplest way to run a one of our notebooks is with a Docker container.
A container provides a self-contained, isolated, and reproducible environment for repetitive experiments.
Docker images are available from the NVIDIA GPU Cloud (NGC).

## 1. Clone the HugeCTR Repository

Use the following command to clone the HugeCTR repository:

```shell
git clone https://github.com/NVIDIA/HugeCTR
```

## 2. Pull the NGC Docker and run it

Pull the container using the following command:

```shell
docker pull nvcr.io/nvidia/merlin/merlin-hugectr:24.06
```

Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command:

   ```shell
   docker run --gpus all --rm -it --cap-add SYS_NICE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -u root -v $(pwd):/HugeCTR -w /HugeCTR --network=host --runtime=nvidia nvcr.io/nvidia/merlin/merlin-hugectr:24.06
   ```  

   > To run the  Sparse Operation Kit notebooks, specify the `nvcr.io/nvidia/merlin/merlin-hugectr:24.06` container.

## 3. Customized Building (Optional)

HugeCTR is already installed in the NGC container. But you can also setup HugeCTR from source to customize the build more. This is useful for developmental purposes.

1. Go to HugeCTR repo and update third party modules

```shell
$ cd HugeCTR
$ git submodule update --init --recursive
```

2. There are options to customize the build using parameters, which are detailed [here](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_contributor_guide.html#build-hugectr-training-container-from-source)
Here are some examples of how you can build HugeCTR using these build options:

```shell
# Example 1
$ mkdir -p build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Target is NVIDIA V100 with all others by default
$ make -j && make install
```
```shell
# Example 2
$ mkdir -p build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" -DENABLE_MULTINODES=ON .. # Target is NVIDIA V100 / A100 with the multi-node mode on.
$ make -j && make install
```

By default, HugeCTR is installed at `/usr/local`. However, you can use `CMAKE_INSTALL_PREFIX` to install HugeCTR to non-default location:

`$ cmake -DCMAKE_INSTALL_PREFIX=/opt/HugeCTR -DSM=70 ..`

Refer to the
> [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development)
> documentation for more details on building HugeCTR From Source

## 4. Start the Jupyter Notebook

1. Start Jupyter using these commands: 

   ```shell
   cd /HugeCTR/notebooks
   jupyter-notebook --allow-root --ip 0.0.0.0 --port 8888 --NotebookApp.token='hugectr'
   ```

2. Connect to your host machine using the 8888 port by accessing its IP address or name from your web browser: `http://[host machine]:8888`

   Use the token available from the output by running the command above to log in. For example:

   `http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`

3. Optional: Import MPI.

   By default, HugeCTR initializes and finalizes MPI when you run the `import hugectr` statement within the NGC Merlin container.
   If you build and install HugeCTR yourself, specify the `ENABLE_MULTINODES=ON` argument when you build.
   See [Build HugeCTR from Source](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_user_guide.html#building-hugectr-from-scratch).

   If your program uses MPI for a reason other than interacting with HugeCTR, initialize MPI with the `from mpi4py import MPI` statement before you import HugeCTR.
   
4. Important Note:

   HugeCTR is written in CUDA/C++ and wrapped to Python using Pybind11. The C++ output will not display in Notebook cells unless you run the Python script in a command line manner.

## Notebook List

The notebooks are located within the container and can be found in the `/HugeCTR/notebooks` directory.

Here's a list of notebooks that you can run:
- [hugectr_e2e_demo_with_nvtabular.ipynb](hugectr_e2e_demo_with_nvtabular.ipynb): Notebook to preprocess data using NVTabular, train the model with HugeCTR.
- [continuous_training.ipynb](continuous_training.ipynb) (deprecated): Notebook to introduce how to deploy continued training with HugeCTR.
- [training_and_inference_with_remote_filesystem.ipynb](training_and_inference_with_remote_filesystem.ipynb): Demonstrates how to train a model with data that is stored in a remote file system such as Hadoop HDFS and AWS S3.

The [multi-modal-data](./multi-modal-data/) series of notebooks demonstrate how to use of multi-modal data such as text and images for the task of movie recommendation.
The notebooks use the Movielens-25M dataset.

Notebooks on the Hierarchical Parameter Server (HPS) are deprecated since v25.03, please check out the version prior to v24.06.

For Sparse Operation Kit notebooks, refer to the [sparse_operation_kit/notebooks/](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/sparse_operation_kit/notebooks) directory of the repository or the [documentation](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/index.html).

## System Specifications

The specifications of the system on which each notebook can run successfully are summarized in the table. The notebooks are verified on the system below but it does not mean the minimum requirements.

| Notebook                                                               | CPU                                                          | GPU                              | #GPUs | Author         |
| ---------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------- | ----- | -------------- |
| [multi-modal-data](multi-modal-data)                                   | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 1     | Vinh Nguyen    |
| [continuous_training.ipynb](continuous_training.ipynb)                 | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 1     | Xiaolei Shi    |
| [training_with_remote_filesystem.ipynb](training_with_remote_filesystem.ipynb) | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 1     | Jerry Shi |
| [hugectr_e2e_demo_with_nvtabular.ipynb](hugectr_e2e_demo_with_nvtabular.ipynb) | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 1     | Jerry Shi |

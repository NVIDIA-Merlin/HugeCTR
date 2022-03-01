# HugeCTR Jupyter demo notebooks
This directory contains a set of Jupyter Notebook demos for HugeCTR.

## Quickstart
The quickest way to run a notebook here is with a docker container, which provides a self-contained, isolated, and reproducible environment for repetitive experiments. HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to use the pre-built Docker image available from the NVIDIA GPU Cloud (NGC). If you want to build the HugeCTR docker image on your own, please refer to [Use Docker Container](../docs/mainpage.md#use-docker-container).

### Pull the NGC Docker
To start the [sparse_operation_kit_demo.ipynb](../sparse_operation_kit/notebooks/sparse_operation_kit_demo.ipynb) notebook, pull this docker image:
```
docker pull nvcr.io/nvidia/merlin/merlin-tensorflow-training:22.02
```

To start the other notebooks, pull the docker image using the following command:
```
docker pull nvcr.io/nvidia/merlin/merlin-training:22.02
```

### Clone the HugeCTR Repository
Use the following command to clone the HugeCTR repository:
```
git clone https://github.com/NVIDIA/HugeCTR
```

### Start the Jupyter Notebook

1. Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command: 
   ```
   docker run --runtime=nvidia --rm -it --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr -p 8888:8888 nvcr.io/nvidia/merlin/merlin-training:22.02
   ```  
   Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command to run [sparse_operation_kit_demo.ipynb](../sparse_operation_kit/notebooks/sparse_operation_kit_demo.ipynb) notebook : 
   ```
   docker run --runtime=nvidia --rm -it --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr -p 8888:8888 nvcr.io/nvstaging/merlin/merlin-tensorflow-training:22.02
   ```

2. Start Jupyter using these commands: 
   ```
   cd /hugectr/notebooks
   jupyter-notebook --allow-root --ip 0.0.0.0 --port 8888 --NotebookApp.token='hugectr'
   ```

3. Connect to your host machine using the 8888 port by accessing its IP address or name from your web browser: `http://[host machine]:8888`

   Use the token available from the output by running the command above to log in. For example:

   `http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`

4. Optional: Import MPI.

   By default, HugeCTR initializes and finalizes MPI when you run the `import hugectr` statement within the NGC Merlin container.
   If you build and install HugeCTR yourself, specify the `ENABLE_MULTINODES=ON` argument when you build.
   See [Build HugeCTR from Source](../docs/hugectr_contributor_guide.md#build-hugectr-from-source).

   If your program uses MPI for a reason other than interacting with HugeCTR, initialize MPI with the `from mpi4py import MPI` statement before you import HugeCTR.
   
5. Important Note:

   HugeCTR is written in CUDA/C++ and wrapped to Python using Pybind11. The C++ output will not display in Notebook cells unless you run the Python script in a command line manner.



## Notebook List
The notebooks are located within the container and can be found here: `/hugectr/notebooks`.

For notebooks of SparseOperationKit, please refer to [sparse_operation_kit/notebooks/](../sparse_operation_kit/notebooks/ReadMe.md)

Here's a list of notebooks that you can run:
- [ecommerce-example.ipynb](ecommerce-example.ipynb): Explains how to train and inference with the eCommerce dataset.
- [movie-lens-example.ipynb](movie-lens-example.ipynb): Explains how to train and inference with the MoveLense dataset.
- [hugectr-criteo.ipynb](hugectr_criteo.ipynb): Explains the usage of HugeCTR Python interface with the Criteo dataset.
- [hugectr2onnx_demo.ipynb](hugectr2onnx_demo.ipynb): Explains how to convert the trained HugeCTR model to ONNX.
- [continuous_training.ipynb](continuous_training.ipynb): Notebook to introduce how to deploy continued training with HugeCTR.
- [hugectr_wdl_prediction.ipynb](hugectr_wdl_prediction.ipynb): Tutorial how to train a wdl model using HugeCTR High-level python API.
- [news-example.ipynb](news-example.ipynb): Tutorial to demonstrate NVTabular for ETL the data and HugeCTR for training Deep Neural Network models on MIND dataset.
- [multi-modal-data](multi-modal-data): This series of notebooks demonstrate the use of multi-modal data (text, image) for the task of movie recommendation, using the Movielens-25M dataset.
- [multi_gpu_offline_inference.ipynb](multi_gpu_offline_inference.ipynb): Explain how to do multi-GPU offline inference with HugeCTR Python APIs.

## System Specifications
The specifications of the system on which each notebook can run successfully are summarized in the table. The notebooks are verified on the system below but it does not mean the minimum requirements.

| Notebook                                                               | CPU                                                          | GPU                              | #GPUs | Author         |
| ---------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------- | ----- | -------------- |
| [multi-modal-data](multi-modal-data)                                   | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 1     | Vinh Nguyen    |
| [continuous_training.ipynb](continuous_training.ipynb)                 | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 1     | Xiaolei Shi    |
| [ecommerce-example.ipynb](ecommerce-example.ipynb)                     | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-16GB<br />16 GB Memory | 8     | Vinh Nguyen    |
| [hugectr2onnx_demo.ipynb](hugectr2onnx_demo.ipynb)                     | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-16GB<br />16 GB Memory | 1     | Kingsley Liu   |
| [hugectr-criteo.ipynb](hugectr_criteo.ipynb)                           | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 1     | Kingsley Liu   |
| [multi_gpu_offline_inference.ipynb](multi_gpu_offline_inference.ipynb) | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 4     | Kingsley Liu   |
| [hugectr_wdl_prediction.ipynb](hugectr_wdl_prediction.ipynb)           | AMD Ryzen 9 3900X 12-Core <br />32 GB Memory                 | GeForce RTX 2080Ti<br />11 GB Memory   | 1       | Yingcan Wei    |
| [movie-lens-example.ipynb](movie-lens-example.ipynb)                   | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 1     | Vinh Nguyen    |
| [news-example.ipynb](news-example.ipynb)                               | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 4     | Ashish Sardana |

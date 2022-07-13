# Hierarchical Parameter Server (HPS) notebooks #
This directory contains a set of Jupyter notebooks that demonstrate how to use HPS in TensorFlow.


## HPS APIs
```python
hierarchical_parameter_server.Init()
```
This methods will initialize the HPS for all the deployed models. It needs to be called only once and must be called before any other HPS APIs.

**Arguments**
* `global_batch_size`: Integer, the global batch size for HPS that is deployed on multiple GPUs.
* `ps_config_file`: String, the JSON configuration file for HPS initialization.

***

```python
hierarchical_parameter_server.LookupLayer()
```
`LookupLayer` is a subclassing `tf.keras.layers.Layer`, and `LookupLayer.call()` takes a Tensor with type int64 containing the ids to be looked up.

**Arguments**
* `model_name`: String, the name of the model that has embedding table(s).
* `table_id`: Integer, the index of the embedding table for the model specified by `model_name`.
* `emb_vec_size`: Integer, the embedding vector size for the embedding table specified by `model_name` and `table_id`.
* `emb_vec_dtype`: tensorflow.python.framework.dtypes.DType, the data type of embedding vectors which must be tf.float32 currently.

## Quickstart

The simplest way to run a one of our notebooks is with a Docker container.
A container provides a self-contained, isolated, and reproducible environment for repetitive experiments.
Docker images are available from the NVIDIA GPU Cloud (NGC).
If you prefer to build the HugeCTR Docker image on your own, refer to [Set Up the Development Environment With Merlin Containers](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#set-up-the-development-environment-with-merlin-containers).

### Pull the NGC Docker

Pull the container using the following command:

```shell
docker pull nvcr.io/nvidia/merlin/merlin-hugectr:22.07
```

### Clone the HugeCTR Repository

Use the following command to clone the HugeCTR repository:

```shell
git clone https://github.com/NVIDIA/HugeCTR
```

### Start the Jupyter Notebook
1. Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command:

   ```shell
   docker run --runtime=nvidia --rm -it --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr -p 8888:8888 nvcr.io/nvidia/merlin/merlin-hugectr:22.07
   ```  

   > To run the  Sparse Operation Kit notebooks, specify the `nvcr.io/nvidia/merlin/merlin-hugectr:22.07` container.

2. Start Jupyter using these commands: 

   ```shell
   cd /hugectr/notebooks
   jupyter-notebook --allow-root --ip 0.0.0.0 --port 8888 --NotebookApp.token='hugectr'
   ```

3. Connect to your host machine using the 8888 port by accessing its IP address or name from your web browser: `http://[host machine]:8888`

   Use the token available from the output by running the command above to log in. For example:

   `http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`


## Notebook List

Here's a list of notebooks that you can run:
- [hierarchical_parameter_server_demo.ipynb](hierarchical_parameter_server_demo.ipynb): Explains how to train with native TF layers and make inference with HPS.


## System Specifications
The specifications of the system on which each notebook can run successfully are summarized in the table. The notebooks are verified on the system below but it does not mean the minimum requirements.

| Notebook                                                                              | CPU                                                          | GPU                              | #GPUs | Author         |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------- | ----- | -------------- |
| [hierarchical_parameter_server_demo.ipynb](hierarchical_parameter_server_demo.ipynb)  | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz<br />512 GB Memory | Tesla V100-SXM2-32GB<br />32 GB Memory | 1     | Kingsley Liu    |

# HugeCTR Jupyter demo notebooks
This directory contains a set of Jupyter Notebook demos for HugeCTR.

## Quickstart
The quickest way to run a notebook here is with a docker container, which provides a self-contained, isolated, and reproducible environment for repetitive experiments. HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to use the pre-built Docker image available from the NVIDIA GPU Cloud (NGC). If you want to build the HugeCTR docker image on your own, please refer to [Use Docker Container](../docs/mainpage.md#use-docker-container).

### Pull the NGC Docker
To start the [embedding_plugin.ipynb](embedding_plugin.ipynb) notebook, pull this docker image:
```
docker pull nvcr.io/nvstaging/merlin/merlin-tensorflow-training:0.5
```

To start the other notebooks, pull the docker image using the following command:
```
docker pull nvcr.io/nvidia/merlin/merlin-training:0.5
```

### Clone the HugeCTR Repository
Use the following command to clone the HugeCTR repository:
```
git clone https://github.com/NVIDIA/HugeCTR
```

### Start the Jupyter Notebook

1. Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command: 
   ```
   docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr -p 8888:8888 nvcr.io/nvidia/merlin/merlin-training:0.5
   ```  
   Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command to run [embedding_plugin.ipynb](embedding_plugin.ipynb) notebook : 
   ```
   docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr -p 8888:8888 nvcr.io/nvstaging/merlin/merlin-tensorflow-training:0.5
   ```

2. Activate the merlin conda environment by running the following command:  
   ```shell.
   source activate merlin
   ```

3. Start Jupyter using these commands: 
   ```
   cd /hugectr/notebooks
   jupyter-notebook --allow-root --ip 0.0.0.0 --port 8888 --NotebookApp.token=’hugectr’
   ```

4. Connect to your host machine using the 8888 port by accessing its IP address or name from your web browser: `http://[host machine]:8888`

   Use the token available from the output by running the command above to log in. For example:

   `http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`


## Notebook List
The notebooks are located within the container and can be found here: `/hugectr/notebooks`.

Here's a list of notebooks that you can run:
- [movie-lens-example.ipynb](movie-lens-example.ipynb): Explains how to train and inference with the MoveLense dataset.
- [embedding_plugin.ipynb](embedding_plugin.ipynb): Explains how to install and use the HugeCTR embedding plugin with Tensorflow.
- [python_interface.ipynb](python_interface.ipynb): Explains how to use the Python interface and the model prefetching feature.
- [hugectr_inference.ipynb](hugectr_inference.ipynb): Explains how to use python interface to predict with a trained model.

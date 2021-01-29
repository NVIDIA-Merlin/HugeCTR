# HugeCTR Jupyter demo notebooks
This directory contains a set of Jupyter Notebook demos for HugeCTR.

## Prerequisites
The quickest way to run a notebook here is with a docker container, which provides a self-contained, isolated, and reproducible environment for repetitive experiments.

### Clone the HugeCTR Repository
Use the following command to clone the HugeCTR repository:
```
git clone https://github.com/NVIDIA/HugeCTR
```
### Build the NVIDIA HugeCTR Container
The followng steps are also outlined in the [README](../README.md#2-build-docker-image-and-hugectr).

1. Inside the root directory of the HugeCTR repository, run the following command:
   ```
   docker build -t hugectr:devel -f ./tools/dockerfiles/train.Dockerfile .
   ```
   **Note**: If you want to try the [**HugeCTR Embedding Plugin for Tensorflow demo**](embedding_plugin.ipynb), run the following command instead:
   ```
   docker build -t hugectr:devel -f ./tools/dockerfiles/plugin-embedding.Dockerfile .
   ```

2. Launch the container in interactive mode (mount the root directory into the container for your convenience) by running this command: 
   ```
   docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr hugectr:devel
   ```

3. Within the docker interactive bash session, build HugeCTR using these commands:
   ```
   mkdir -p build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Asuming the target GPU is Volta-generation GPU, such as the Tesla V100
   make -j
   ```
4. Within the docker interactive bash session, build HugeCTR using these commands:
   ```
   mkdir -p build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Asuming the target GPU is Volta-generation GPU, such as the Tesla V100
   make -j
   ```

5. Install and start Jupyter using these commands: 
   ```
   pip3 install --upgrade notebook
   cd /hugectr
   jupyter-notebook --allow-root --ip 0.0.0.0 --port 8888
   ```

6. Connect to your host machine using the 8888 port by accessing its IP address or name from your web browser: `http://[host machine]:8888`

   Use the token available from the output by running the command above to log in. For example:

   `http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`


## Notebook List
The notebooks are located within the container and can be found here: `/hugectr/notebooks`.

Here's a list of notebooks that you can run:
- [movie-lens-example.ipynb](movie-lens-example.ipynb): Explains how to train and inference with the MoveLense dataset.
- [embedding_plugin.ipynb](embedding_plugin.ipynb): Explains how to install and use the HugeCTR embedding plugin with Tensorflow.
- [python_interface.ipynb](python_interface.ipynb): Explains how to use the Python interface and the model prefetching feature.

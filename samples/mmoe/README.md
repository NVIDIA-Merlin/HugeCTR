# MMoE SAMPLE #
The purpose of this sample is to demonstrate how to build and train a [Multi-gate Mixture of Experts (MMoE) model](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) with HugeCTR.

## Set Up the HugeCTR Docker Environment ##
You can set up the HugeCTR Docker environment by doing one of the following:
- [Pull the NGC Docker](#pull-the-ngc-docker)
- [Build the HugeCTR Docker Container on Your Own](#build-the-hugectr-docker-container-on-your-own)

### Pull the NGC Docker ###
HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to pull the pre-built Docker image, which is available on the NVIDIA GPU Cloud (NGC). This method provides a self-contained, isolated, and reproducible environment for repetitive experiments.

1. Pull the HugeCTR NGC Docker by running the following command:
   ```bash
   $ docker pull nvcr.io/nvidia/merlin/merlin-hugectr:22.09
   ```
2. Launch the container in interactive mode with the HugeCTR root directory mounted into the container by running the following command:
   ```bash
   $ docker run --gpus=all --rm -it --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-hugectr:22.09
   ```

### Build the HugeCTR Docker Container on Your Own ###
Please refer to [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development) to build on your own and set up the Docker container. Make sure that HugeCTR is built and installed to the system path `/usr/local/hugectr` within the Docker container. Launch the container in interactive mode in the same manner as above, and then set the `PYTHONPATH` environment variable inside the Docker container using the following command:
```shell
$ export PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH
```
## Preparing your dataset for HugeCTR ##
If you have a multi-label dataset that you would like to train with this HugeCTR MMoE sample, first preprocess the data to a format that is supported in HugeCTR.  The [preprocessing scripts](tools/criteo_script) for the Criteo 1TB dataset outline how to preprocess a dataset to an accepted format. In a future release we will provide sample datasets that can be used as a template for formatting your multi-task dataset for HugeCTR MMoE training.

## Train and validate the MMoE model ##
Once you have your dataset formatted into the Norm format the local directory, update the filename in `mmoe.py` and update other parameters as needed to fit your dataset.  Details on selecting the correct parameters are available in the [HugeCTR User guide](docs/hugectr_user_guide.md).  Once updated, train your MMoE model by running:
``` shell
$ python mmoe.py
```

If you are using the Parquet format dataset, update the `mmoe_parquet.py` file and run it using:
``` shell
$ python mmoe_parquet.py
```

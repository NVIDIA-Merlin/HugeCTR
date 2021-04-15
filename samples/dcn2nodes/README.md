# DCN MULTI-NODES SAMPLE #
The purpose of this sample is to build and train the [Deep & Cross Network](https://arxiv.org/pdf/1708.05123.pdf) with multi-node enabled within HugeCTR.

## Set Up the HugeCTR Docker Environment ##
You can set up the HugeCTR Docker environment by doing one of the following:
- [Pull the NGC Docker](#pull-the-ngc-docker)
- [Build the HugeCTR Docker Container on Your Own](#build-the-hugectr-docker-container-on-your-own)

### Pull the NGC Docker ###
HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to pull the pre-built Docker image, which is available on the NVIDIA GPU Cloud (NGC). This method provides a self-contained, isolated, and reproducible environment for repetitive experiments.

1. Pull the HugeCTR NGC Docker by running the following command:
   ```bash
   $ docker pull nvcr.io/nvidia/merlin/merlin-inference:0.5
   ```
2. Launch the container in interactive mode with the HugeCTR root directory mounted into the container by running the following command:
   ```bash
   $ docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-inference:0.5
   ```

### Build the HugeCTR Docker Container on Your Own ###
If you want to build the HugeCTR Docker container on your own, refer to [Build HugeCTR Docker Containers](../../tools/dockerfiles#build-container-for-model-training) and [Use the Docker Container](../docs/mainpage.md#use-docker-container). For more information about building HugeCTR with multi-node enabled, see [Build with Multi-Nodes Training Supported](../docs/mainpage.md#build-with-multi-nodes-training-supported). You should make sure that HugeCTR is built and installed in `/usr/local/hugectr` within the Docker container. You can launch the container in interactive mode in the same manner as shown above.

## Download the Dataset ##
Go [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) and download one of the dataset files into the "${project_root}/tools" directory. 

As an alternative, you can run the following command:
```
$ cd ${project_root}/tools
$ wget http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_1.gz
```

**NOTE**: Replace `1` with a value from [0, 23] to use a different day.

During preprocessing, the amount of data, which is used to speed up the preprocessing, fill missing values, and remove the feature values that are considered rare, is further reduced.

## Preprocess the Dataset ##
When running this sample, the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) is used. The dataset contains 24 files in which each file corresponds to one day of data. To reduce preprocessing time, only one file is used. Each sample consists of a label (0 if the ad wasn't clicked and 1 if the ad was clicked) and 39 features (13 integer features and 26 categorical features). The dataset is also missing numerous values across the feature columns, which should be preprocessed accordingly.

### Preprocess the Dataset Through Pandas ####
To preprocess the dataset through Pandas, run the following command:
```shell
$ bash preprocess.sh 1 criteo_data pandas 1 0
```

**IMPORTANT NOTES**: 
- The first argument represents the dataset postfix. For instance, if `day_1` is used, the postfix is `1`.
- The second argument `criteo_data` is where the preprocessed data is stored. You may want to change it in cases where multiple datasets are generated concurrently. If you change it, `source` and `eval_source` in your JSON configuration file must be changed as well.
- The fourth argument (the one after `pandas`) represents if the normalization is applied to dense features (1=ON, 0=OFF).
- The last argument determines if feature crossing should be applied. It must remain set `0` (OFF).

## Train with HugeCTR ##
If the gossip communication library is used, a plan file must be generated first as shown below. If the NCCL communication library is used, there is no need to generate a plan file and you can proceed to step 2. 

1. Run `huge_ctr`.
   ```shell
   $ mpirun --bind-to none huge_ctr --train /samples/dcn2nodes/dcn8l8gpu2nodes.json
   ```

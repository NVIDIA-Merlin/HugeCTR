# CRITEO CTR SAMPLE #
The purpose of this sample is to demonstrate the basic usage of `huge_ctr`.

## Set Up the HugeCTR Docker Environment ##
You can set up the HugeCTR Docker environment by doing one of the following:
- [Pull the NGC Docker](#pull-the-ngc-docker)
- [Build the HugeCTR Docker Container on Your Own](#build-the-hugectr-docker-container-on-your-own)

### Pull the NGC Docker ###
HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to pull the pre-built Docker image, which is available on the NVIDIA GPU Cloud (NGC). This method provides a self-contained, isolated, and reproducible environment for repetitive experiments.

1. Pull the HugeCTR NGC Docker by running the following command:
   ```bash
   $ docker pull nvcr.io/nvidia/merlin/merlin-training:0.5
   ```
2. Launch the container in interactive mode with the HugeCTR root directory mounted into the container by running the following command:
   ```bash
   $ docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-training:0.5
   ```  
3. Activate the merlin conda environment by running the following command:  
   ```shell.
   source activate merlin
   ```

### Build the HugeCTR Docker Container on Your Own ###
If you want to build the HugeCTR Docker container on your own, refer to [Build HugeCTR Docker Containers](../../tools/dockerfiles#build-container-for-model-training) and [Use the Docker Container](../docs/mainpage.md#use-docker-container).

You should make sure that HugeCTR is built and installed in `/usr/local/hugectr` within the Docker container. You can launch the container in interactive mode in the same manner as shown above.

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

After you've downloaded the dataset, you can use one of the following methods to prepare the dataset for HugeCTR training:
- [Perl](#preprocess-the-dataset-through-perl)
- [NVTabular](#preprocess-the-dataset-through-nvtabular)

### Preprocess the Dataset Through Perl ###
To preprocess the dataset through Perl, run the following command:
```shell
$ bash preprocess.sh 1 criteo_data perl 1
```
**IMPORTANT NOTES**: 
- The first argument represents the dataset postfix.  For example, if `day_1` is used, the postfix is `1`.
- The second argument `criteo_data` is where the preprocessed data is stored. You may want to change it in cases where multiple datasets are generated concurrently. If you change it, `source` and `eval_source` in your JSON configuration file must be changed as well.
- The last argument represents the number of slots used in this sample.

### Preprocess the Dataset Through NVTabular ###
HugeCTR supports data processing through NVTabular. Make sure that the NVTabular Docker environment has been set up successfully. For more information, see [NVTAbular github](https://github.com/NVIDIA/NVTabular). Ensure that you're using the latest version of NVTabular and mount the HugeCTR ${project_root} volume into the NVTabular Docker.

1. Run NVTabular Docker and execute the following preprocessing command:
   ```shell
   $ bash preprocess.sh 1 criteo_data nvt 1 1 0
   ```
2. Exit from the NVTabular Docker environment and launch the HugeCTR Docker in interactive mode with the HugeCTR root directory mounted into the  
   container.

**IMPORTANT NOTES**: 
- The first and second arguments are as the same as Perl's as shown above.
- If you want to generate binary data using the `Norm` data format instead of the `Parquet` data format, set the fourth argument (the one after `nvt`) to `0`. Generating binary data using the `Norm` data format can take much longer than it does when using the `Parquet` data format because of the additional conversion process. Use the NVTabular binary mode if you encounter an issue with Pandas mode.
- The fifth argument (the one after `nvt`)  must be set to `1`.
- The last argument determines whether the feature crossing should be applied (1=ON, 0=OFF). It must remain set to `0`.

## Train with HugeCTR ##
Run the following command after preprocessing the dataset through Perl:
```shell
$ huge_ctr --train ../samples/criteo/criteo.json
```

Run one of the following commands after preprocessing the dataset through NVTabular using either the Parquet or Binary output:

**Parquet Output**
```shell
$ huge_ctr --train ../samples/criteo/criteo_parquet.json
```

**Binary Output**
```shell
$ huge_ctr --train ../samples/criteo/criteo_bin.json
```

**NOTE**: If you want to generate binary data using the `Norm` data format instead of the `Parquet` data format, set the fourth argument (the one after `nvt`) to `0`. Generating binary data using the `Norm` data format can take much longer than it does when using the `Parquet` data format because of the additional conversion process. Use the NVTabular binary mode if you encounter an issue with Pandas mode.

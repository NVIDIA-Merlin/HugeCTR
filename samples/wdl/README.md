# WDL CTR SAMPLE #
A sample of building and training Wide & Deep Network with HugeCTR [(link)](https://arxiv.org/abs/1606.07792).

## Setup the HugeCTR Docker Environment ##
The quickest way to run the sample here is with a docker container, which provides a self-contained, isolated, and reproducible environment for repetitive experiments. HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to use the pre-built Docker image available from the NVIDIA GPU Cloud (NGC). If you want to build the HugeCTR docker image on your own, please refer to [Use Docker Container](../docs/mainpage.md#use-docker-container).

You can choose either to pull the NGC docker or to build on your own.

#### Pull the NGC Docker ####
Pull the HugeCTR NGC docker using this command:
```bash
$ docker pull nvcr.io/nvidia/hugectr:v3.0
```
Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command:
```bash
$ docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/hugectr:v3.0
```

#### Build on Your Own ####
Please refer to [Use Docker Container](../docs/mainpage.md#use-docker-container) to build on your own and set up the docker container. Please make sure that HugeCTR is built and installed to the system path `/usr/local/hugectr` within the docker container. Please launch the container in interactive mode in the same manner as above.

## Dataset and Preprocess ##
In running this sample, [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) is used.
The dataset contains 24 files, each of which corresponds to one day of data.
To spend less time on preprocessing, we use only one of them.
Each sample consists of a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
The dataset also has the significant amounts of missing values across the feature columns, which should be preprocessed accordingly.

#### Download the Dataset ####

Go to [this link](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/),
and download one of 24 files into the directory "${project_root}/tools", 
or execute the following command:
```
$ cd ${project_root}/tools
$ wget http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_1.gz
```
- **NOTE**: Replace 1 with a value from [0, 23] to use a different day.

In preprocessing, we will further reduce the amounts of data to speedup the preprocessing, fill missing values, remove the feature values whose occurrences are very rare, etc.
Please choose one of the following two methods to make the dataset ready for HugeCTR training.

#### Preprocessing by Pandas ####
```shell
$ bash preprocess.sh 1 criteo_data pandas 1 1
```
- **NOTE**: The first argument represents the dataset postfix.  For instance, if `day_1` is used, it is 1.
- **NOTE**: the second argument `criteo_data` is where the preprocessed data is stored.
You may want to change it in case where multiple datasets for different purposes are generated concurrently.
If you change it, `source` and `eval_source` in your JSON config file must be changed as well.
- **NOTE**: the fourth arguement (one after `pandas`) represents if the normalization is applied to dense features (1=ON, 0=OFF).
- **NOTE**: the last argument decides if the feature crossing is applied (1=ON, 0=OFF).
It must remains 0 unless the sample is not `wdl`.

#### Preprocessing by NVTabular ####

HugeCTR supports data processing by NVTabular since version 2.2.1.
Please make sure NVTabular docker environment has been set up successfully according to [NVTAbular github](https://github.com/NVIDIA/NVTabular).
Make sure to use the latest version of NVTabular,
and mount HugeCTR ${project_root} volume to NVTabular docker.
Run NVTabular docker and execute the following preprocessing commands:
```shell
$ bash preprocess.sh 1 criteo_data nvt 1 0 1 # parquet output
```
Or
```shell
$ bash preprocess.sh 1 criteo_data nvt 0 0 1 # nvt binary output
```
- **NOTE**: The first and second arguments are as the same as Pandas's (see above).
- **NOTE**: If you want to generate a binary data in `Norm` format data, instead of the Parquet format data, set the fourth argument (one after `nvt`) to 0. It can take much longer than the Parquet mode becuase of the additional conversion process.
Otherwise, a Parquet dataset is generated. Use this NVTabular binary mode if you encounter an  issue with the Pandas mode.
- **NOTE**: the fifth argument must be set to 1 for `criteo` sample. Otherwise, it is 0.
- **NOTE**: the last argument decides if the feature crossing is applied (1=ON, 0=OFF).
It must remains 0 unless the sample is not `wdl`.

Exit from the NVTabular docker environment and then run HugeCTR docker with interaction mode under home directory again.


## Training with HugeCTR ##

#### After Pandas Preprocessing ####
```shell
$ huge_ctr --train ../samples/wdl/wdl.json
```

#### After NVTabular Preprocessing ####

Parquet output
```shell
$ huge_ctr --train ../samples/wdl/wdl_parquet.json
```
Binary output (See NOTE above)
```shell
$ huge_ctr --train ../samples/wdl/wdl_bin.json
```

## More Explaination on The Network Configuration ##
```
 "sparse": [
	{
          "top": "wide_data",
          "type": "DistributedSlot",
          "max_feature_num_per_sample": 30,
          "slot_num": 1
        },
  }
```
```
  {
      "name": "sparse_embedding2",
      "type": "DistributedSlotSparseEmbeddingHash",
      "bottom": "wide_data",
      "top": "sparse_embedding2",
      "sparse_embedding_hparam": {
        "max_vocabulary_size_per_gpu": 2322444,
        "embedding_vec_size": 1,
        "combiner": 0
      }
    },
```
As wide model in WDL is Logistic Regression, the `slot_num` in data layer and `embedding_vec_size` in Embedding layer should be always 1. 

P = sigmoid(b0 + b1*x1 + b2*x2 + b3*x3)

In this situation, b is the parameters in our embedding above and x is the input vector (multi-hot). A sum combiner in a slot in embedding can be used to simulate `b1*x1 + b2*x2 + b3*x3`

```
    {
      "name": "reshape1",
      "type": "Reshape",
      "bottom": "sparse_embedding1",
      "top": "reshape1",
      "leading_dim": 416
    },
```

The Reshape layer after embedding usually has `leading_dim` = `slot_num`*`embedding_vec_size`, which means a concatenation of the category features.
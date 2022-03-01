# WDL CTR SAMPLE #
The purpose of this sample is to demonstrate how to build and train a [Wide & Deep Network](https://arxiv.org/abs/1606.07792) with HugeCTR.

## Set Up the HugeCTR Docker Environment ##
You can set up the HugeCTR Docker environment by doing one of the following:
- [Pull the NGC Docker](#pull-the-ngc-docker)
- [Build the HugeCTR Docker Container on Your Own](#build-the-hugectr-docker-container-on-your-own)

### Pull the NGC Docker ###
HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to pull the pre-built Docker image, which is available on the NVIDIA GPU Cloud (NGC). This method provides a self-contained, isolated, and reproducible environment for repetitive experiments.

1. Pull the HugeCTR NGC Docker by running the following command:
   ```bash
   $ docker pull nvcr.io/nvidia/merlin/merlin-training:22.02
   ```
2. Launch the container in interactive mode with the HugeCTR root directory mounted into the container by running the following command:
   ```bash
   $ docker run --gpus=all --rm -it --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-training:22.02
   ```

### Build the HugeCTR Docker Container on Your Own ###
If you want to build the HugeCTR Docker container on your own, please refer to [How to Start Your Development](../../docs/hugectr_contributor_guide.md#how-to-start-your-development).

You should make sure that HugeCTR is built and installed in `/usr/local/hugectr` within the Docker container. You can launch the container in interactive mode in the same manner as shown above, and then set the `PYTHONPATH` environment variable inside the Docker container using the following command:
```shell
$ export PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH
``` 

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

After you've downloaded the dataset, you can use one of the following methods to prepare the dataset for HugeCTR trainings:
- [Pandas](#preprocess-the-dataset-through-pandas)
- [NVTabular](#preprocess-the-dataset-through-nvtabular)

### Preprocess the Dataset Through Pandas ###
To preprocess the dataset through Pandas, run the following command:
```shell
$ bash preprocess.sh 1 criteo_data pandas 1 1
```

**IMPORTANT NOTES**: 
- The first argument represents the dataset postfix.  For instance, if `day_1` is used, the postfix is `1`.
- The second argument, `criteo_data`, is where the preprocessed data is stored. You may want to change it in cases where multiple datasets are generated concurrently. If you change it, `source` and `eval_source` in your JSON configuration file must be changed as well.
- The fourth argument (the one after `pandas`) represents if the normalization is applied to dense features (1=ON, 0=OFF).
- The last argument decides if the feature crossing is applied (1=ON, 0=OFF).

### Preprocess the Dataset Through NVTabular ###
HugeCTR supports data processing through NVTabular. Make sure that the NVTabular Docker environment has been set up successfully. For more information, see [NVTAbular github](https://github.com/NVIDIA/NVTabular). Ensure that you're using the latest version of NVTabular and mount the HugeCTR ${project_root} volume into the NVTabular Docker.

Execute one of the following preprocessing commands:
   ```shell
   $ bash preprocess.sh 1 criteo_data nvt 1 0 1 # parquet output
   ```

   ```shell
   $ bash preprocess.sh 1 criteo_data nvt 0 0 1 # nvt binary output
   ```

**IMPORTANT NOTES**: 
- The first and second arguments are the same as Pandas's, as shown above.
- If you want to generate binary data using the `Norm` data format instead of the `Parquet` data format, set the fourth argument (the one after `nvt`) to `0`. Generating binary data using the `Norm` data format can take much longer than it does when using the `Parquet` data format because of the additional conversion process. Use the NVTabular binary mode if you encounter an issue with Pandas mode.
- The last argument determines whether the feature crossing should be applied (1=ON, 0=OFF).

## Train with HugeCTR ##
Run the following command after preprocessing the dataset through Pandas:
```shell
$ python3 ../samples/wdl/wdl.py
```

To train your model with HugeCTR on an 8-GPU machine such as DGX, run the following command after preprocessing the dataset through Pandas:
```shell
$ python3 ../samples/wdl/wdl_8gpu.py
```

Run one of the following commands after preprocessing the dataset through NVTabular using either the Parquet or Binary output:

**Parquet Output**
```shell
$ python3 ../samples/wdl/wdl_parquet.py
```

**Binary Output**
```shell
$ python3 ../samples/wdl/wdl_bin.py
```

**NOTE**: If you want to generate binary data using the `Norm` data format instead of the `Parquet` data format, set the fourth argument (the one after `nvt`) to `0`. Generating binary data using the `Norm` data format can take much longer than it does when using the `Parquet` data format because of the additional conversion process. Use the NVTabular binary mode if you encounter an issue with Pandas mode.

## Additional Details About The Model Graph ##
```
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Distributed, 30, 2, 1),
                        hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Distributed, 30, 1, 26)],
                        sparse_names = ["wide_data", "deep_data"]))
```
```
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 69,
                            embedding_vec_size = 1,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding2",
                            bottom_name = "wide_data",
                            optimizer = optimizer))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 1074,
                            embedding_vec_size = 16,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "deep_data",
                            optimizer = optimizer))
```
Because the wide model in WDL is Logistic Regression, the `slot_num` in the data layer and `embedding_vec_size` in the Embedding layer should always be `1`.

P = sigmoid(b0 + b1*x1 + b2*x2 + b3*x3)

In this case, `b` represents the parameters in our embedding above and `x` represents the input vector (multi-hot). A sum combiner in a slot in the embedding can be used to simulate `b1*x1 + b2*x2 + b3*x3`

```
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding1"],
                            top_names = ["reshape1"],
                            leading_dim=416))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding2"],
                            top_names = ["reshape2"],
                            leading_dim=1))
```

The Reshape layer after the embedding usually has `leading_dim` = `slot_num`*`embedding_vec_size`, which means a concatenation of the category features.

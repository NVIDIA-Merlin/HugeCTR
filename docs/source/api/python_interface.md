# HugeCTR Python Interface

## About the HugeCTR Python Interface

As a recommendation system domain specific framework, HugeCTR has a set of high level abstracted Python Interface which includes training API and inference API. Users only need to focus on algorithm design, the training and inference jobs can be automatically deployed on the specific hardware topology in the optimized manner. From version 3.1, users can complete the process of training and inference without manually writing JSON configuration files. All supported functionalities have been wrapped into high-level Python APIs. Meanwhile, the low-level training API is maintained for users who want to have precise control of each training iteration and each evaluation step. Still, the high-level training API is friendly to users who are already familiar with other deep learning frameworks like Keras and it is worthwhile to switch to it from low-level training API. Please refer to [HugeCTR Python Interface Notebook](../notebooks/hugectr_criteo.ipynb) to get familiar with the workflow of HugeCTR training and inference. Meanwhile we have a lot of samples for demonstration in the [`samples`](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/samples) directory of the HugeCTR repository.

## High-level Training API

For HugeCTR high-level training API, the core data structures are `Solver`, `EmbeddingTrainingCacheParams`, `DataReaderParams`, `OptParamsPy`, `Input`, `SparseEmbedding`, `DenseLayer` and `Model`. You can create a `Model` instance with `Solver`, `EmbeddingTrainingCacheParams`, `DataReaderParams` and `OptParamsPy` instances, and then add instances of `Input`, `SparseEmbedding` or `DenseLayer` to it. After compiling the model with the `Model.compile()` method, you can start the epoch mode or non-epoch mode training by simply calling the `Model.fit()` method. Moreover, the `Model.summary()` method gives you an overview of the model structure. We also provide some other methods, such as saving the model graph to a JSON file, constructing the model graph based on the saved JSON file, loading model weights and optimizer status, etc.

### Solver

#### CreateSolver method

```python
hugectr.CreateSolver()
```

`CreateSolver` returns an `Solver` object according to the custom argument values，which specify the training resources.

**Arguments**
* `model_name`: String, the name of the model. The default value is empty string. If you want to dump the model graph and save the model weights for inference, a unique value should be specified for each model that needs to be deployed.

* `seed`: A random seed to be specified. The default value is 0.

* `lr_policy`: The learning rate policy which suppots only fixed. The default value is `LrPolicy_t.fixed`.

* `lr`: The learning rate, which is also the base learning rate for the learning rate scheduler. The default value is 0.001.

* `warmup_steps`: The warmup steps for the internal learning rate scheduler within Model instance. The default value is 1.

* `decay_start`: The step at which the learning rate decay starts for the internal learning rate scheduler within Model instance. The default value is 0.

* `decay_steps`: The number of steps of the learning rate decay for the internal learning rate scheduler within Model instance. The default value is 1.

* `decay_power`: The power of the learning rate decay for the internal learning rate scheduler within Model instance. The default value is 2.

* `end_lr`: The final learning rate for the internal learning rate scheduler within Model instance. The default value is 0. Please refer to [SGD Optimizer and Learning Rate Scheduling](hugectr_core_features.md#sgd-optimizer-and-learning-rate-scheduling) if you want to get detailed information about LearningRateScheduler.

* `max_eval_batches`: Maximum number of batches used in evaluation. It is recommended that the number is equal to or bigger than the actual number of bathces in the evaluation dataset. The default value is 100.

* `batchsize_eval`: Minibatch size used in evaluation. The default value is 2048. **Note that batchsize here is the global batch size across gpus and nodes, not per worker batch size.**

* `batchsize`: Minibatch size used in training. The default value is 2048. **Note that batchsize here is the global batch size across gpus and nodes , not per worker batch size.**

* `vvgpu`: GPU indices used in the training process, which has two levels. For example: [[0,1],[1,2]] indicates that two physical nodes (each physical node can have multiple NUMA nodes) are used. In the first node, GPUs 0 and 1 are used while GPUs 1 and 2 are used for the second node. It is also possible to specify non-continuous GPU indices such as [0, 2, 4, 7]. The default value is [[0]].

* `repeat_dataset`: Whether to repeat the dataset for training. If the value is `True`, non-epoch mode training will be employed. Otherwise, epoch mode training will be adopted. The default value is `True`.

* `use_mixed_precision`: Whether to enable mixed precision training. The default value is `False`.

* `enable_tf32_compute`: If you want to accelerate FP32 matrix multiplications within the FullyConnectedLayer and InteractionLayer, set this value to `True`. The default value is `False`.

* `scaler`: The scaler to be used when mixed precision training is enabled. Only 128, 256, 512, and 1024 scalers are supported for mixed precision training. The default value is 1.0, which corresponds to no mixed precision training.

* `metrics_spec`: Map of enabled evaluation metrics. You can use either AUC, AverageLoss, HitRate, or any combination of them. For AUC, you can set its threshold, such as {MetricsType.AUC: 0.8025}, so that the training terminates when it reaches that threshold. The default value is {MetricsType.AUC: 1.0}. Multiple metrics can be specified in one job. For example: metrics_spec = {hugectr.MetricsType.HitRate: 0.8, hugectr.MetricsType.AverageLoss:0.0, hugectr.MetricsType.AUC: 1.0})

* `i64_input_key`: If your dataset format is `Norm`, you can choose the data type of each input key. For the `Parquet` format dataset generated by NVTabular, only I64 is allowed. For the `Raw` dataset format, only I32 is allowed. Set this value to `True` when you need to use I64 input key. The default value is `False`.

* `use_algorithm_search`: Whether to use algorithm search for cublasGemmEx within the FullyConnectedLayer. The default value is `True`.

* `use_cuda_graph`: Whether to enable cuda graph for dense network forward and backward propagation. The default value is `True`.

* `device_layout`: The layout of the device map for the resource manager. The supported options include `DeviceLayout.LocalFirst` and `DeviceLayout.NODE_FIRST`. If `DeviceLayout.NODE_FIRST` is employed, all nodes should have same number of devices. This argument is restricted to MLPerf use and the default value is `DeviceLayout.LocalFirst`.

* `use_holistic_cuda_graph`: If this option is enabled, everything inside a training iteration is packed into a CUDA Graph. This option works only if `use_cuda_graph` is turned off and `use_overlapped_pipeline` is turned on. This argument is restricted to MLPerf use and the default value is `False`.

* `use_overlapped_pipeline`: If this option is turned on, the bottom MLP computation will be overlapped with the hybrid embedding computation. This argument is restricted to MLPerf use and the default value is `False`.

* `all_reduce_algo`: The algorithm to be used for all reduce. The supported options include `AllReduceAlgo.OneShot` and `AllReduceAlgo.NCCL`. This argument is restricted to MLPerf use and the default value is `AllReduceAlgo.NCCL`. When you are doing multi-node training, `AllReduceAlgo.OneShot` will require RDMA support while `AllReduceAlgo.NCCL` can run on both RDMA and non-RDMA hardware.

* `grouped_all_reduce`: Whether to use grouped all reduce. This argument is restricted to MLPerf use and the default value is `False`.

* `num_iterations_statistics`: The number of batches that are used in performing the statistics. This argument is restricted to MLPerf use and the default value is 20.

* `is_dlrm`: A global flag to specify whether to apply all the MLPerf optimizations for DLRM sample. The MLPerf specific options will be valid only if this flag is set `True`. The default value is `False`.

Example:
```python
solver = hugectr.CreateSolver(max_eval_batches = 300,
                              batchsize_eval = 16384,
                              batchsize = 16384,
                              lr = 0.001,
                              vvgpu = [[0]],
                              repeat_dataset = True,
                              i64_input_key = True)
```

***

#### CreateETC method

```python
hugectr.CreateETC()
```

`CreateETC` should **only** be called when using the [Embedding Training Cache](../hugectr_embedding_training_cache.md) (ETC) feature. It returns a `EmbeddingTrainingCacheParams` object that specifies the parameters for initializing a `EmbeddingTrainingCache` instance.

**Arguments**
* `ps_types`: A list specifies types of [parameter servers](hugectr_embedding_training_cache.md#parameter-server-in-etc) (PS) of each embedding table. Available PS choices for embeddings are:
  * [`hugectr.TrainPSType_t.Staged`](hugectr_embedding_training_cache.md#staged-host-memory-parameter-server)
    * The whole embedding table will be loaded into the host memory in the initialization stage.
    * It requires the size of host memory should be large enough to hold the embedding table along with the optimizer states (if any).
    * *`Staged` type offers better loading and dumping bandwidth than the `Cached` PS.*
  * [`hugectr.TrainPSType_t.Cached`](hugectr_embedding_training_cache.md#cached-host-memory-parameter-server)
    * A sub-portion of the embedding table will be dynamically cached in the host memory, and it adopts a runtime eviction/insertion mechanism to update the cached table.
    * The size of the cached table is configurable, which can be substantially smaller than the size of the embedding table stored in the SSD or various kinds of filesystems. E.g., embedding table size (1 TB) v.s. cache size (100 GB).
    * The bandwidth of `Cached` PS is mainly affected by the hit rate. If the hit rate is 100 %, its bandwidth tends to the `Staged` PS; Otherwise, if the hit rate is 0 %, the bandwidth equals the random-accessing bandwidth of SSD.

* `sparse_models`: A path list of embedding table(s). If the provided path points to an existing table, this table will be used for incremental training. Otherwise, the newly generated table will be written into this path after training.

* `local_paths`: A path list for storing the temporary embedding table. Its length should be equal to the number of MPI ranks. Each entry in this list should be a path pointing to the local SSD of this node.

  *This entry is only required when there is `hugectr.TrainPSType_t.Cached` in `ps_types`.*

* `hcache_configs`: A path list of the configurations of `Cached` PS. Please check [Cached-PS Configuration](hugectr_embedding_training_cache.md#cached-ps-configuration) for more descriptions.
  * If only one configuration is provided, it will be used for all `Cached` PS.
  * Otherwise, you need to provide one configuration for each `Cached` PS. And the ith configuration in `hcache_configs` will be used for the ith occurrence of `Cached` PS in `ps_types`.

  *This entry is only required when there is `hugectr.TrainPSType_t.Cached` in `ps_types`.*

**Note that the `Staged` and `Cached` PS can be used together for a model with more than one embedding tables.**

Example usage of the `CreateETC()` API can be found in [Configuration](hugectr_embedding_training_cache.md#configuration).

For the usage of the ETC feature in real cases, please check the [HugeCTR Continuous Training](../notebooks/continuous_training.ipynb) notebook.

### AsyncParam

#### AsyncParam class

```python
hugectr.AsyncParam()
```

`AsyncParam` specifies the parameters related to async raw data reader, which can be used to initialize `DataReaderParams` instance. It is restricted to MLPerf use.

**Arguments**
* `num_threads`: Integer, the number of the data reading threads, should be at least 1 per GPU。 This argument is restricted to MLPerf use and there is NO default value.

* `num_batches_per_thread`: Integer,  the number of the batches each data reader thread works on simultaneously, typically 2-4. This argument is restricted to MLPerf use and there is NO default value.

* `io_block_size`: Integer, the size of individual IO requests, the value 512000 should work in most cases. This argument is restricted to MLPerf use and there is NO default value.

* `io_depth`: Integer, the size of the asynchronous IO queue, the value 4 should work in most cases. This argument is restricted to MLPerf use and there is NO default value.

* `io_alignment`: Integer, the byte alignment of IO requests, the value 512 should work in most cases. This argument is restricted to MLPerf use and there is NO default value.

* `shuffle`: Boolean, if this option is enabled, the order in which the batches are fed into training will be randomized. This argument is restricted to MLPerf use and there is NO default value.

* `aligned_type`: The supported types include `hugectr.Alignment_t.Auto` and `hugectr.Alignment_t.Non`. If `hugectr.Alignment_t.Auto` is chosen,  the dimension of dense input will be padded to an 8-aligned value. This argument is restricted to MLPerf use and there is NO default value.

Example:
```python
async_param = hugectr.AsyncParam(32, 4, 716800, 2, 512, True, hugectr.Alignment_t.Non)
```

### HybridEmbeddingParam

#### HybridEmbeddingParam class

```python
hugectr.HybridEmbeddingParam()
```

`HybridEmbeddingParam` specifies the parameters related to hybrid embedding, which can be used to initialize `SparseEmbedding` instance. It is restricted to MLPerf use.

**Arguments**
* `max_num_frequent_categories`: Integer, the maximum number of frequent categories in unit of batch size. This argument is restricted to MLPerf use and there is NO default value.

* `max_num_infrequent_samples`: Integer, the maximum number of infrequent samples in unit of batch size. This argument is restricted to MLPerf use and there is NO default value.

* `p_dup_max`: Float, the maximum probability that the category appears more than once within the gpu-batch. This way of determining the number of frequent categories is used in single-node or NVLink connected systems only. This argument is restricted to MLPerf use and there is NO default value.

* `max_all_reduce_bandwidth`: Float, the bandwidth of the all reduce. This argument is restricted to MLPerf use and there is NO default value.

* `max_all_to_all_bandwidth`: Float, the bandwidth of the all-to-all. This argument is restricted to MLPerf use and there is NO default value.

* `efficiency_bandwidth_ratio`: Float, this argument is used in combination with `max_all_reduce_bandwidth` and `max_all_to_all_bandwidth` to determine the optimal threshold for number of frequent categories. This way of determining the frequent categories is used for multi node only. This argument is restricted to MLPerf use and there is NO default value.

* `communication_type`: The type of communication that is being used. The supported types include `CommunicationType.IB_NVLink`, `CommunicationType.IB_NVLink_Hier` and `CommunicationType.NVLink_SingleNode`. This argument is restricted to MLPerf use and there is NO default value.

* `hybrid_embedding_type`: The type of hybrid embedding, which supports only `HybridEmbeddingType.Distributed` for now. This argument is restricted to MLPerf use and there is NO default value.

Example:
```python
hybrid_embedding_param = hugectr.HybridEmbeddingParam(2, -1, 0.01, 1.3e11, 1.9e11, 1.0,
                                                    hugectr.CommunicationType.IB_NVLink_Hier,
                                                    hugectr.HybridEmbeddingType.Distributed))
```

### DataReaderParams

#### DataReaderParams class

```bash
hugectr.DataReaderParams()
```

`DataReaderParams` specifies the parameters related to the data reader. HugeCTR currently supports three dataset formats, i.e., [Norm](#norm), [Raw](#raw) and [Parquet](#parquet). An `DataReaderParams` instance is required to initialize the `Model` instance.

**Arguments**
* `data_reader_type`: The type of the data reader which should be consistent with the dataset format. The supported types include `hugectr.DataReaderType_t.Norm`, `hugectr.DataReaderType_t.Raw` and `hugectr.DataReaderType_t.Parquet` and `DataReaderType_t.RawAsync`. The type `DataReaderType_t.RawAsync` is valid only if `is_dlrm` is set `True` within `CreateSolver`. There is NO default value and it should be specified by users.

* `source`: List[str] or String, the training dataset source. For Norm or Parquet dataset, it should be the file list of training data, e.g., `source = "file_list.txt"`. For Raw dataset, it should be a single training file, e.g., `source = "train_data.bin"`. When using embedding training cache, it can be specified with several file lists, e.g., `source = ["file_list.1.txt", "file_list.2.txt"]`. There is NO default value and it should be specified by users.

* `keyset`: List[str] or String, the keyset files. This argument will ONLY be valid when using embedding training cache and it should be corresponding to the `source`. For example, we can specify `source = ["file_list.1.txt", "file_list.2.txt"]` and `source = ["file_list.1.keyset", "file_list.2.keyset"]`, which have a one-to-one correspondence.

* `eval_source`: String, the evaluation dataset source. For Norm or Parquet dataset, it should be the file list of evaluation data. For Raw dataset, it should be a single evaluation file. There is NO default value and it should be specified by users.

* `check_type`: The data error detection mechanism. The supported types include `hugectr.Check_t.Sum` (CheckSum) and `hugectr.Check_t.Non` (no detection). There is NO default value and it should be specified by users.

* `cache_eval_data`: Integer, the cache size of evaluation data on device, set this parameter greater than zero to restrict the memory that will be used. The default value is 0.

* `num_samples`: Integer, the number of samples in the traning dataset. This is ONLY valid for Raw dataset. The default value is 0.

* `eval_num_samples`: Integer, the number of samples in the evaluation dataset. This is ONLY valid for Raw dataset. The default value is 0.

* `float_label_dense`: Boolean, this is valid only for the Raw dataset format. If its value is set to `True`, the label and dense features for each sample are interpreted as float values. Otherwise, they are read as integer values while the dense features are preprocessed with log(dense[i] + 1.f). The default value is `True`.

* `num_workers`: Integer, the number of data reader workers that concurrently load data. You can empirically decide the best one based on your dataset, training environment. The default value is 12.

* `slot_size_array`: List[int], the cardinality array of input features. It should be consistent with that of the sparse input. We requires this argument for Parquet format data and RawAsync format when you want to add offset to input key. The default value is an empty list.

* `async_param`: AsyncParam, the parameters for async raw data reader. This argument is restricted to MLPerf use.

### Dataset formats

We support the following dataset formats within our `DataReaderParams`.

* [Norm](#norm)
* [Raw](#raw)
* [Parquet](#parquet)

<img src ="/user_guide_src/dataset_format.png" width="80%" align="center"/>

<div align=center>Fig. 1: (a) Norm (b) Raw (c) Parquet Dataset Formats</div>

<br></br>

#### Norm

To maximize the data loading performance and minimize the storage, the Norm dataset format consists of a collection of binary data files and an ASCII formatted file list. The model file should specify the file name of the training and testing (evaluation) set, maximum elements (key) in a sample, and the label dimensions as shown in Fig. 1 (a).

##### Data Files

A data file is the minimum reading granularity for a reading thread, so at least 10 files in each file list are required to achieve the best performance. A data file consists of a header and actual tabular data.

Header Definition:

```c
typedef struct DataSetHeader_ {
  long long error_check;        // 0: no error check; 1: check_num
  long long number_of_records;  // the number of samples in this data file
  long long label_dim;          // dimension of label
  long long dense_dim;          // dimension of dense feature
  long long slot_num;           // slot_num for each embedding
  long long reserved[3];        // reserved for future use
} DataSetHeader;
```

Data Definition (each sample):

```c
typedef struct Data_ {
  int length;                   // bytes in this sample (optional: only in check_sum mode )
  float label[label_dim];
  float dense[dense_dim];
  Slot slots[slot_num];
  char checkbits;                // checkbit for this sample (optional: only in checksum mode)
} Data;

typedef struct Slot_ {
  int nnz;
  unsigned int*  keys; // changeable to `long long` with `"input_key_type"` in `solver` object of the configuration file.
} Slot;
```

The Data field often has a lot of samples. Each sample starts with the labels formatted as integers and followed by `nnz` (number of nonzero) with the input key using the long long (or unsigned int) format as shown in Fig. 1 (a).

The input keys for categorical are distributed to the slots with no overlap allowed. For example: `slot[0] = {0,10,32,45}, slot[1] = {1,2,5,67}`. If there is any overlap, it will cause an undefined behavior. For example, given `slot[0] = {0,10,32,45}, slot[1] = {1,10,5,67}`, the table looking up the `10` key will produce different results based on how the slots are assigned to the GPUs.

##### File List

The first line of a file list should be the number of data files in the dataset with the paths to those files listed below as shown here:

```shell
$ cat simple_sparse_embedding_file_list.txt
10
./simple_sparse_embedding/simple_sparse_embedding0.data
./simple_sparse_embedding/simple_sparse_embedding1.data
./simple_sparse_embedding/simple_sparse_embedding2.data
./simple_sparse_embedding/simple_sparse_embedding3.data
./simple_sparse_embedding/simple_sparse_embedding4.data
./simple_sparse_embedding/simple_sparse_embedding5.data
./simple_sparse_embedding/simple_sparse_embedding6.data
./simple_sparse_embedding/simple_sparse_embedding7.data
./simple_sparse_embedding/simple_sparse_embedding8.data
./simple_sparse_embedding/simple_sparse_embedding9.data
```

Example:

```python
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ["./wdl_norm/file_list.txt"],
                                  eval_source = "./wdl_norm/file_list_test.txt",
                                  check_type = hugectr.Check_t.Sum)
```

#### Raw

The Raw dataset format is different from the Norm dataset format in that the training data appears in one binary file using int32. Fig. 1 (b) shows the structure of a Raw dataset sample.

**NOTE**: Only one-hot data is accepted with this format.

The Raw dataset format can be used with embedding type LocalizedSlotSparseEmbeddingOneHot only.

Example:

```python
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Raw,
                                  source = ["./wdl_raw/train_data.bin"],
                                  eval_source = "./wdl_raw/validation_data.bin",
                                  check_type = hugectr.Check_t.Sum)
```

#### Parquet

Parquet is a column-oriented, open source, and free data format. It is available to any project in the Apache Hadoop ecosystem. To reduce the file size, it supports compression and encoding. Fig. 1 (c) shows an example Parquet dataset. For additional information, see the [parquet documentation](https://parquet.apache.org/docs/).

Please note the following:

* Nested column types are not currently supported in the Parquet data loader.
* Any missing values in a column are not allowed.
* Like the Norm dataset format, the label and dense feature columns should use the float format.
* The Slot feature columns should use the Int64 format.
* The data columns within the Parquet file can be arranged in any order.
* To obtain the required information from all the rows in each parquet file and column index mapping for each label, dense (numerical), and slot (categorical) feature, a separate `_metadata.json` file is required.

Example `_metadata.json` file:

```json
{
   "file_stats":[
      {
         "file_name":"file0.parquet",
         "num_rows":409600
      },
      {
         "file_name":"file1.parquet",
         "num_rows":409600
      }
   ],
   "cats":[
      {
         "col_name":"C1",
         "index":4
      },
      {
         "col_name":"C2",
         "index":5
      },
      {
         "col_name":"C3",
         "index":6
      },
      {
         "col_name":"C4",
         "index":7
      }
   ],
   "conts":[
      {
         "col_name":"I1",
         "index":1
      },
      {
         "col_name":"I2",
         "index":2
      },
      {
         "col_name":"I3",
         "index":3
      }
   ],
   "labels":[
      {
         "col_name":"label",
         "index":0
      }
   ]
}
```

```python
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                  source = ["./parquet_data/train/_file_list.txt"],
                                  eval_source = "./parquet_data/val/_file_list.txt",
                                  check_type = hugectr.Check_t.Non,
                                  slot_size_array = [10000, 50000, 20000, 300])
```

We provide an option to add offset for each slot by specifying `slot_size_array`. `slot_size_array` is an array whose length is equal to the number of slots. To avoid duplicate keys after adding offset, we need to ensure that the key range of the i-th slot is between 0 and slot_size_array[i]. We will do the offset in this way: for i-th slot key, we add it with offset slot_size_array[0] + slot_size_array[1] + ... + slot_size_array[i - 1]. In the configuration snippet noted above, for the 0th slot, offset 0 will be added. For the 1st slot, offset 10000 will be added. And for the third slot, offset 60000 will be added. The length of `slot_size_array` should be equal to the length of `"cats"` entry in `_metadata.json`.


The `_metadata.json` is generated by [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) preprocessing and reside in the same folder of the file list. Basically, it contain four entries of `"file_stats"` (file statistics), `"cats"` (categorical columns), `"conts"` (continuous columns), and `"labels"` (label columns). The `"col_name"` and `"index"` in `_metadata.json` indicate the name and the index of a specific column in the parquet data frame. You can also edit the generated `_metadata.json` to only read the desired columns for model training. For example, you can modify the above `_metadata.json` and change the configuration correspondingly:

Example `_metadata.json` file after edits:

```json
{
   "file_stats":[
      {
         "file_name":"file0.parquet",
         "num_rows":409600
      },
      {
         "file_name":"file1.parquet",
         "num_rows":409600
      }
   ],
   "cats":[
      {
         "col_name":"C2",
         "index":5
      },
      {
         "col_name":"C4",
         "index":7
      }
   ],
   "conts":[
      {
         "col_name":"I1",
         "index":1
      },
      {
         "col_name":"I3",
         "index":3
      }
   ],
   "labels":[
      {
         "col_name":"label",
         "index":0
      }
   ]
}
```

```python
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                  source = ["./parquet_data/train/_file_list.txt"],
                                  eval_source = "./parquet_data/val/_file_list.txt",
                                  check_type = hugectr.Check_t.Non,
                                  slot_size_array = [50000, 300])
```

### OptParamsPy

#### CreateOptimizer method

```python
hugectr.CreateOptimizer()
```

`CreateOptimizer` returns an `OptParamsPy` object according to the custom argument values，which specify the optimizer type and the corresponding hyperparameters. The `OptParamsPy` object will be used to initialize the `Model` instance and it applies to the weights of dense layers. Sparse embedding layers which do not have a specified optimizer will adopt this optimizer as well. Please **NOTE** that the hyperparameters should be configured meticulously when mixed precision training is employed, e.g., the `epsilon` value for the `Adam` optimizer should be set larger.

The embedding update supports three algorithms specified with `update_type`:

* `Local` (default value): The optimizer will only update the hot columns (embedding vectors which is hit in this iteration of training) of an embedding in each iteration.
* `Global`: The optimizer will update all the columns. The embedding update type takes longer than the other embedding update types.
* `LazyGlobal`: The optimizer will only update the hot columns of an embedding in each iteration while using different semantics from the *local* and *global* updates.

**Arguments**
* `optimizer_type`: The optimizer type to be used. The supported types include `hugectr.Optimizer_t.Adam`, `hugectr.Optimizer_t.MomentumSGD`, `hugectr.Optimizer_t.Nesterov` and `hugectr.Optimizer_t.SGD`, `hugectr.Optimizer_t.Adagrad`. The default value is `hugectr.Optimizer_t.Adam`.

* `update_type`: The update type for the embedding. The supported types include `hugectr.Update_t.Global`, `hugectr.Update_t.Local`, and `hugectr.Update_t.LazyGlobal`(Adam only). The default value is `hugectr.Update_t.Global`.

* `beta1`: The `beta1` value when using Adam optimizer. The default value is 0.9.

* `beta2`: The `beta2` value when using Adam optimizer. The default value is 0.999.

* `epsilon`: The `epsilon` value when using Adam optimizer. This argument should be well configured when mixed precision training is employed. The default value is 1e-7.

* `momentum_factor`: The `momentum_factor` value when using MomentumSGD or Nesterov optimizer. The default value is 0.

* `atomic_update`: Whether to employ atomic update when using SGD optimizer. The default value is True.

Example:

```python
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                    update_type = hugectr.Update_t.Global,
                                    beta1 = 0.9,
                                    beta2 = 0.999,
                                    epsilon = 0.0000001)
```

## Layers

There are three major kinds of `layer` in HugeCTR:

* [Input](#input-layer)
* [Sparse Embedding](#sparseembedding)
* [Dense](#denselayer)

Please refer to [hugectr_layer_book](./hugectr_layer_book.md) for detail guides on how to use different layer types.

### Input Layer

```python
hugectr.Input()
```

`Input`specifies the parameters related to the data input. An `Input` instance should be added to the Model instance first so that the following `SparseEmbedding` and `DenseLayer` instances can access the inputs with their specified names. Please refer to [Input Detail](./hugectr_layer_book.md#input-layer) if you want to get detailed information about Input.

**Arguments**
* `label_dim`: Integer, the label dimension. 1 implies it is a binary label. For example, if an item is clicked or not. Optionally a list of Integers for multi-label data. There is NO default value and it should be specified by users.

* `label_name`: String, the name of the label tensor to be referenced by following layers. Optionally a list of Strings for multi-label data.  If multiple labels given, the number must match label_dim. There is NO default value and it should be specified by users.

* `dense_dim`: Integer, the number of dense (or continuous) features. If there is no dense feature, set it to 0. There is NO default value and it should be specified by users.

* `dense_name`: Integer, the name of the dense input tensor to be referenced by following layers. There is NO default value and it should be specified by users.

* `data_reader_sparse_param_array`: List[hugectr.DataReaderSparseParam], the list of the sparse parameters for categorical inputs. Each `DataReaderSparseParam` instance should be constructed with  `sparse_name`, `nnz_per_slot`, `is_fixed_length` and `slot_num`.
  * `sparse_name` is the name of the sparse input tensors to be referenced by following layers. There is NO default value and it should be specified by users.
  * `nnz_per_slot` is the maximum number of features for each slot for the specified spare input. The `nnz_per_slot` can be an `int` which means average nnz per slot so the maximum number of features per sample should be `nnz_per_slot * slot_num`. Or you can use List[int] to initialize `nnz_per_slot`, then the maximum number of features per sample should be `sum(nnz_per_slot)` and in this case, the length of the array `nnz_per_slot` should be the same with `slot_num`.
  * `is_fixed_length` is used to identify whether categorical inputs has the same length for each slot among all samples. If different samples have the same number of features for each slot, then user can set `is_fixed_length = True` and HugeCTR can use this information to reduce data transferring time.
  * `slot_num` specifies the number of slots used for this sparse input in the dataset. **Note:** if multiple `DataReaderSparseParam` are specified there's no overlap between any pair of `DataReaderSparseParam`. e.g. in our [wdl sample](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/samples/wdl/wdl.py), we have 27 slots in total; we specified the first slot as "wide_data" and the next 26 slots as "deep_data".

### SparseEmbedding

```python
hugectr.SparseEmbedding()
```

`SparseEmbedding` specifies the parameters related to the sparse embedding layer. One or several `SparseEmbedding` layers should be added to the Model instance after `Input` and before `DenseLayer`. Please refer to [SparseEmbedding Detail](./hugectr_layer_book.md#sparse-embedding) if you want to get detailed information about SparseEmbedding.

**Arguments**
* `embedding_type`: The embedding type to be used. The supported types include `hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash`, `hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash`, `hugectr.Embedding_t.LocalizedSlotSparseEmbeddingOneHot` and `hugectr.Embedding_t.HybridSparseEmbedding`. The type `Embedding_t.HybridSparseEmbedding` is valid only if `is_dlrm` is set `True` within `CreateSolver` and `data_reader_type` is specified as `DataReaderType_t.RawAsync` within `DataReaderParams`. There is NO default value and it should be specified by users.

* `workspace_size_per_gpu_in_mb`: Integer, the workspace memory size in megabyte per GPU. This workspace memory must be big enough to hold all the embedding vocabulary and its corresponding optimizer state used during the training and evaluation. There is NO default value and it should be specified by users. To understand how to set this value, please refer [How to set workspace_size_per_gpu_in_mb and slot_size_array](/QAList.md#how-to-set-workspace-size-per-gpu-in-mb-and-slot-size-array).

* `embedding_vec_size`: Integer, the embedding vector size. There is NO default value and it should be specified by users.

* `combiner`: String, the intra-slot reduction operation, currently `sum` or `mean` are supported. There is NO default value and it should be specified by users.

* `sparse_embedding_name`: String, the name of the sparse embedding tensor to be referenced by following layers. There is NO default value and it should be specified by users.

* `bottom_name`: String, the number of the bottom tensor to be consumed by this sparse embedding layer. Please note that it should be a predefined sparse input name. There is NO default value and it should be specified by users.

* `slot_size_array`: List[int], the cardinality array of input features. It should be consistent with that of the sparse input. This parameter can be used in `LocalizedSlotSparseEmbeddingHash`, `LocalizedSlotSparseEmbeddingOneHot` and `HybridSparseEmbedding`. The meaning of `slot_size_array` is varied based on different embedding type. There is NO default value and it should be specified by users. Please refer [How to set workspace_size_per_gpu_in_mb and slot_size_array](/QAList.md#how-to-set-workspace-size-per-gpu-in-mb-and-slot-size-array)

* `optimizer`: OptParamsPy, the optimizer dedicated to this sparse embedding layer. If the user does not specify the optimizer for the sparse embedding, it will adopt the same optimizer as dense layers.

* `hybrid_embedding_param`: HybridEmbeddingParam, the parameters for hybrid embedding. This argument is restricted to MLPerf use.

### DenseLayer

```python
hugectr.DenseLayer()
```

`DenseLayer` specifies the parameters related to the dense layer or the loss function. HugeCTR currently supports multiple dense layers and loss functions, Please refer to [DenseLayer Detail](./hugectr_layer_book.md#dense-layers) if you want to get detailed information about dense layers. Please **NOTE** that the final sigmoid function is fused with the loss function to better utilize memory bandwidth.

**Arguments**
* `layer_type`: The layer type to be used. The supported types include `hugectr.Layer_t.Add`, `hugectr.Layer_t.BatchNorm`, `hugectr.Layer_t.Cast`, `hugectr.Layer_t.Concat`, `hugectr.Layer_t.Dropout`, `hugectr.Layer_t.ELU`, `hugectr.Layer_t.FmOrder2`, `hugectr.Layer_t.FusedInnerProduct`, `hugectr.Layer_t.InnerProduct`, `hugectr.Layer_t.Interaction`, `hugectr.Layer_t.MultiCross`, `hugectr.Layer_t.ReLU`, `hugectr.Layer_t.ReduceSum`, `hugectr.Layer_t.Reshape`, `hugectr.Layer_t.Sigmoid`, `hugectr.Layer_t.Slice`, `hugectr.Layer_t.WeightMultiply`, `hugectr.ElementWiseMultiply`, `hugectr.Layer_t.GRU`, `hugectr.Layer_t.Scale`, `hugectr.Layer_t.FusedReshapeConcat`, `hugectr.Layer_t.FusedReshapeConcatGeneral`, `hugectr.Layer_t.Softmax`, `hugectr.Layer_t.PReLU_Dice`, `hugectr.Layer_t.ReduceMean`, `hugectr.Layer_t.Sub`, `hugectr.Layer_t.Gather`, `hugectr.Layer_t.BinaryCrossEntropyLoss`, `hugectr.Layer_t.CrossEntropyLoss` and `hugectr.Layer_t.MultiCrossEntropyLoss`. There is NO default value and it should be specified by users.

* `bottom_names`: List[str], the list of bottom tensor names to be consumed by this dense layer. Each name in the list should be the predefined tensor name. There is NO default value and it should be specified by users.

* `top_names`: List[str], the list of top tensor names, which specify the output tensors of this dense layer. There is NO default value and it should be specified by users.

* `factor`: Float, exponential average factor such as runningMean = runningMean*(1-factor) + newMean*factor for the `BatchNorm` layer. The default value is 1.

* `eps`: Float, epsilon value used in the batch normalization formula for the `BatchNorm` layer. The default value is 1e-5.

* `gamma_init_type`: Specifies how to initialize the gamma (or scale) array for the `BatchNorm` layer. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.

* `beta_init_type`: Specifies how to initialize the beta (or offset) array for the `BatchNorm` layer. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.

* `dropout_rate`: Float, The dropout rate to be used for the `Dropout` layer. It should be between 0 and 1. Setting it to 1 indicates that there is no dropped element at all. The default value is 0.5.

* `elu_alpha`: Float, the scalar that decides the value where this `ELU` function saturates for negative values. The default value is 1.

* `num_output`: Integer, the number of output elements for the `InnerProduct` or `FusedInnerProduct` layer. The default value is 1.

* `weight_init_type`: Specifies how to initialize the weight array for the `InnerProduct`, `FusedInnerProduct`, `MultiCross` or `WeightMultiply` layer. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.

* `bias_init_type`: Specifies how to initialize the bias array for the `InnerProduct`, `FusedInnerProduct` or `MultiCross` layer. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.

* `num_layers`: Integer, the Number of cross layers for the `MultiCross` layer. It should be set as a positive number if you want to use the cross network. The default value is 0.

* `leading_dim`: Integer, the innermost dimension of the output tensor for the `Reshape` layer. It must be the multiple of the total number of input elements. The default value is 1.

* `selected`: Boolean, whether to use the selected mode for the `Reshape` layer. The default value is False.

* `selected_slots`: List[int], the selected slots for the `Reshape` layer. It will be ignored if `selected` is False. The default value is [].

* `ranges`: List[Tuple[int, int]], used for the `Slice` layer. A list of tuples in which each one represents a range in the input tensor to generate the corresponding output tensor. For example, (2, 8) indicates that 8 elements starting from the second element in the input tensor are used to create an output tensor. The number of tuples corresponds to the number of output tensors. Ranges are allowed to overlap unless it is a reverse or negative range. The default value is [].

* `weight_dims`: List[int], the shape of the weight matrix (slot_dim, vec_dim) where vec_dim corresponds to the latent vector length for the `WeightMultiply` layer. It should be set correctly if you want to employ the weight multiplication. The default value is [].

* `out_dim`: Integer, the output vector size for the `FmOrder2` layer. It should be set as a positive number if your want to use factorization machine. The default value is 0.

* `axis`: Integer, the dimension to reduce for the `ReduceSum` layer. If the input is N-dimensional, 0 <= axis < N. The default value is 1.

* `time_step`: Integer, the secondary dimension of the output tensor of the `Reshape` layer. It has to be used with `leading_dim` to define 3D output tensor for `Reshape` layer. The default value is 0.

* `batchsize`: Integer, the require information of the `GRU` layer. The default value is 1.

* `SeqLength`: Integer, the require information of the `GRU` layer. The default value is 1.

* `vector_size`: Integer, the require information of the `GRU` layer. The default value is 1.

* `indices`: List[int], a list of indices of the `Gather` layer to specific the extract slice of the input tensor. The default value is [].

* `target_weight_vec`: List[float], the target weight vector for the `MultiCrossEntropyLoss` layer. The default value is [].

* `use_regularizer`: Boolean, whether to use the regularizer for the `BinaryCrossEntropyLoss`, `CrossEntropyLoss` or `MultiCrossEntropyLoss` layer. The default value is False.

* `regularizer_type`: The regularizer type for the `BinaryCrossEntropyLoss`, `CrossEntropyLoss` or `MultiCrossEntropyLoss` layer. The supported types include `hugectr.Regularizer_t.L1` and `hugectr.Regularizer_t.L2`. It will be ignored if `use_regularizer` is False. The default value is `hugectr.Regularizer_t.L1`.

* `lambda`: Float, the lambda value of the regularization term for the `BinaryCrossEntropyLoss`, `CrossEntropyLoss` or `MultiCrossEntropyLoss` layer. It will be ignored if `use_regularizer` is False. The default value is 0.

* `pos_type`: The position type of `FusedInnerProduct` layer. The supported types include `FcPosition_t.Head`, `FcPosition_t.Body`, `FcPosition_t.Tail`, `FcPosition_t.Isolated` and `FcPosition_t.Non`. If the type `FcPosition_t.Non` is specified, the general `FusedFullyConnectedLayer` will be used internally. Otherwise, the MLPerf specific `FusedReluBiasFullyConnectedLayer` will be employed and it requires `is_dlrm` to be `True` within `CreateSolver`. The default value is `FcPosition_t.Non`.

* `act_type`: The activation type of `FusedInnerProduct` layer. The supported types include `Activation_t.Relu` and `Activation_t.Non`. This argument is valid only if `is_dlrm` is set `True` within `CreateSolver` and `layer_type` is specified as `hugectr.Layer_t.FusedInnerProduct`. Besides, `Activation_t.Non` can only be used together with `FcPosition_t.Tail`. The default value is `Activation_t.Relu`.

### GroupDenseLayer

```python
hugectr.GroupDenseLayer()
```

`GroupDenseLayer` specifies the parameters related to a group of dense layers. HugeCTR currently supports only `GroupFusedInnerProduct`, which is comprised of multiple `FusedInnerProduct` layers. Please **NOTE** that the `FusedInnerProduct` layer only supports fp16.

**Arguments**
* `group_layer_type`: The layer type to be used. There is only one supported type, i.e., `hugectr.GroupLayer_t.GroupFusedInnerProduct`. There is NO default value and it should be specified by users.

* `bottom_name_list`: List[str], the list of bottom tensor names for the first dense layer in this group. Currently, the `FusedInnerProduct` layer at the head position can take one or two input tensors. There is NO default value and it should be specified by users.

* `top_name_list`: List[str], the list of top tensor names of each dense layer in the group. There should be only one name for each layer. There is NO default value and it should be specified by users.

* `num_outputs`: List[Integer], the number of output elements for each `FusedInnerProduct` layer in the group. There is NO default value and it should be specified by users.

* `last_act_type`: The activation type of the last `FusedInnerProduct` layer in the group. The supported types include `Activation_t.Relu` and `Activation_t.Non`. Except the last layer, the activation type of the other `FusedInnerProduct` layers in the group must be and will be automatically set as `Activation_t.Relu`, which do not allow any configurations. The default value is `Activation_t.Relu`.

**NOTE**: There should be at least two layers in the group, and the size of `top_name_list` and `num_outputs` should both be equal to the number of layers.

Example:

```python
model.add(hugectr.GroupDenseLayer(group_layer_type = hugectr.GroupLayer_t.GroupFusedInnerProduct,
                                  bottom_name = ["dense"],
                                  top_name_list = ["fc1", "fc2", "fc3"],
                                  num_outputs = [1024, 512, 256],
                                  last_act_type = hugectr.Activation_t.Relu))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,
                            bottom_names = ["fc3","sparse_embedding1"],
                            top_names = ["interaction1", "interaction1_grad"]))
model.add(hugectr.GroupDenseLayer(group_layer_type = hugectr.GroupLayer_t.GroupFusedInnerProduct,
                            bottom_name_list = ["interaction1", "interaction1_grad"],
                            top_name_list = ["fc4", "fc5", "fc6", "fc7", "fc8"],
                            num_outputs = [1024, 1024, 512, 256, 1],
                            last_act_type = hugectr.Activation_t.Non))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc8", "label"],
                            top_names = ["loss"]))
```

### Model

#### Model class

```python
hugectr.Model()
```

`Model` groups data input, embeddings and dense network into an object with traning features. The construction of `Model` requires a `Solver` instance , a `DataReaderParams` instance, an `OptParamsPy` instance and a `EmbeddingTrainingCacheParams` instance (optional).

**Arguments**
* `solver`: A hugectr.Solver object, the solver configuration for the model.

* `reader_params`: A hugectr.DataReaderParams object, the data reader configuration for the model.

* `opt_params`: A hugectr.OptParamsPy object, the optimizer configuration for the model.

* `etc`: A hugectr.EmbeddingTrainingCacheParams object, the embedding training cache configuration for the model. This argument should **only** be provided when using the embedding training cache feature.

***

#### add method

```python
hugectr.Model.add()
```

The `add` method of Model adds an instance of Input, SparseEmbedding, DenseLayer or GroupDenseLayer to the created Model object. Typically, a Model object is comprised of one Input, several SparseEmbedding and a series of DenseLayer instances. Please note that the loss function for HugeCTR model training is taken as a DenseLayer instance.

**Arguments**
* `input` or `sparse_embedding` or `dense_layer`: This method is an overloaded method that can accept `hugectr.Input`, `hugectr.SparseEmbedding`, `hugectr.DenseLayer` or `hugectr.GroupDenseLayer` as an argument. It allows the users to construct their model flexibly without the JSON configuration file.

***

#### compile method

```python
hugectr.Model.compile()
```

This method requires no extra arguments. It allocates the internal buffer and initializes the model. For multi-task models, can optionally take two arguments.

**Arguments**
* `loss_names`: List of Strings, the list of loss label names to provide weights for.

* `loss_weights`: List of Floats, the weights to be assigned to each loss label.  Number of elements must match the number of loss_names.

***

#### fit method

```python
hugectr.Model.fit()
```

It trains the model for a fixed number of epochs (epoch mode) or iterations (non-epoch mode). You can switch the mode of training through different configurations. To use epoch mode training, `repeat_dataset` within `CreateSolver()` should be set as `False` and `num_epochs` within `Model.fit()` should be set as a positive number. To use non-epoch mode training, `repeat_dataset` within `CreateSolver()` should be set as `True` and `max_iter` within `Model.fit()` should be set as a positive number.

**Arguments**
* `num_epochs`: Integer, the number of epochs for epoch mode training. It will be ignored if `repeat_dataset` is `True`. The default value is 0.

* `max_iter`: Integer, the maximum iteration of non-epoch mode training. It will be ignored if `repeat_dataset` is `False`. The default value is 2000.

* `display`: Integer, the interval of iterations at which the training loss will be displayed. The default value is 200.

* `eval_interval`: Integer, the interval of iterations at which the evaluation will be executed. The default value is 1000.

* `snapshot`: Integer, the interval of iterations at which the snapshot model weights and optimizer states will be saved to files. This argument is invalid when embedding training cache is being used, which means no model parameters will be saved. The default value is 10000.

* `snapshot_prefix`: String, the prefix of the file names for the saved model weights and optimizer states. This argument is invalid when embedding training cache is being used, which means no model parameters will be saved. The default value is `''`.

* `data_source_params`: Optional, hugectr.data.`DataSourceParams`, a struct to specify the file system and paths to use while dumping the models.

***

#### summary method

```python
hugectr.Model.summary()
```

This method takes no extra arguments and prints a string summary of the model. Users can have an overview of the model structure with this method.

***

#### graph_to_json method

```python
hugectr.Model.graph_to_json()
```

This method saves the model graph to a JSON file, which can be used for continuous training and inference.

**Arguments**
* `graph_config_file`: The JSON file to which the model graph will be saved. There is NO default value and it should be specified by users.

***

#### construct_from_json method

```python
hugectr.Model.construct_from_json()
```

This method constructs the model graph from a saved JSON file, which is useful for continuous training and fine-tune.

**Arguments**
* `graph_config_file`: The saved JSON file from which the model graph will be constructed. There is NO default value and it should be specified by users.

* `include_dense_network`: Boolean, whether to include the dense network when constructing the model graph. If it is `True`, the whole model graph will be constructed, then both saved sparse model weights and dense model weights can be loaded. If it is `False`, only the sparse embedding layers will be constructed and the corresponding sparse model weights can be loaded, which enables users to construct a new dense network on top of that. Please NOTE that the HugeCTR layers are organized by names and you can check the input name, output name and output shape and of the added layers with `Model.summary()`. There is NO default value and it should be specified by users.

***

#### load_dense_weights method

```python
hugectr.Model.load_dense_weights()
```

This method load the dense weights from the saved dense model file.

**Arguments**
* `dense_model_file`: String, the saved dense model file from which the dense weights will be loaded. There is NO default value and it should be specified by users.

* `data_source_params`: Optional, hugectr.data.`DataSourceParams`, a struct to specify the file system and paths to use while loading the dense model. If `data_source_params.use_hdfs` is set to `False`, `dense_model_file` will be used as the path.

***

#### load_dense_optimizer_states method

```python
hugectr.Model.load_dense_optimizer_states()
```

This method load the dense optimizer states from the saved dense optimizer states file.

**Arguments**
* `dense_opt_states_file`: String, the saved dense optimizer states file from which the dense optimizer states will be loaded. There is NO default value and it should be specified by users.

* `data_source_params`: Optional, hugectr.data.`DataSourceParams`, a struct to specify the file system and paths to use while loading the dense optimizer states. If `data_source_params.use_hdfs` is set to `False`, `dense_opt_states_file` will be used as the path.

***

#### load_sparse_weights method

```python
hugectr.Model.load_sparse_weights()
```

This method load the sparse weights from the saved sparse embedding files.

Implementation Ⅰ

**Arguments**
* `sparse_embedding_files`: List[str], the sparse embedding files from which the sparse weights will be loaded. The number of files should equal to that of the sparse embedding layers in the model. There is NO default value and it should be specified by users.

* `data_source_params`: Optional, hugectr.data.`DataSourceParams`, a struct to specify the file system and paths to use while loading the sparse models. If `data_source_params.use_hdfs` is set to `False`, `sparse_embedding_files` will be used as the path.

Implementation Ⅱ

**Arguments**
* `sparse_embedding_files_map`: Dict[str, str], the sparse embedding file will be loaded by the embedding layer with the specified sparse embedding name. There is NO default value and it should be specified by users.

Example:

```python
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
                            workspace_size_per_gpu_in_mb = 23,
                            embedding_vec_size = 1,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding2",
                            bottom_name = "wide_data",
                            optimizer = optimizer))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
                            workspace_size_per_gpu_in_mb = 358,
                            embedding_vec_size = 16,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "deep_data",
                            optimizer = optimizer))
# ...
model.load_sparse_weights(["wdl_0_sparse_4000.model", "wdl_1_sparse_4000.model"]) # load models for both embedding layers
model.load_sparse_weights({"sparse_embedding1": "wdl_1_sparse_4000.model"}) # or load the model for one embedding layer
```

***

#### load_sparse_optimizer_states method

```python
hugectr.Model.load_sparse_optimizer_states()
```

This method load the sparse optimizer states from the saved sparse optimizer states files.

Implementation Ⅰ

**Arguments**
* `sparse_opt_states_files`: List[str], the sparse optimizer states files from which the sparse optimizer states will be loaded. The number of files should equal to that of the sparse embedding layers in the model. There is NO default value and it should be specified by users.

* `data_source_params`: Optional, hugectr.data.`DataSourceParams`, a struct to specify the file system and paths to use while loading the sparse optimizer states. If `data_source_params.use_hdfs` is set to `False`, `sparse_opt_states_files` will be used as the path.

Implementation Ⅱ

**Arguments**
* `sparse_opt_states_files_map`: Dict[str, str], the sparse optimizer states file will be loaded by the embedding layer with the specified sparse embedding name. There is NO default value and it should be specified by users.

***

#### freeze_dense method

```python
hugectr.Model.freeze_dense()
```

This method takes no extra arguments and freezes the dense weights of the model. Users can use this method when they want to fine-tune the sparse weights.

***

#### freeze_embedding method

```python
hugectr.Model.freeze_embedding()
```

Implementation Ⅰ: freeze the weights of all the embedding layers.
This method takes no extra arguments and freezes the sparse weights of the model. Users can use this method when they only want to train the dense weights.

Implementation Ⅱ: freeze the weights of a specific embedding layer. Please refer to Section 3.4 of [HugeCTR Criteo Notebook](../notebooks/hugectr_criteo.ipynb) for the usage.

**Arguments**
* `embedding_name`: String, the name of the embedding layer.

Example:

```python
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
                            workspace_size_per_gpu_in_mb = 23,
                            embedding_vec_size = 1,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding2",
                            bottom_name = "wide_data",
                            optimizer = optimizer))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
                            workspace_size_per_gpu_in_mb = 358,
                            embedding_vec_size = 16,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "deep_data",
                            optimizer = optimizer))
# ...
model.freeze_embedding() # freeze all the embedding layers
model.freeze_embedding("sparse_embedding1") # or free a specific embedding layer
```

***

#### unfreeze_dense method

```python
hugectr.Model.unfreeze_dense()
```

This method takes no extra arguments and unfreezes the dense weights of the model.

***

#### unfreeze_embedding method

```python
hugectr.Model.unfreeze_embedding()
```

Implementation Ⅰ: unfreeze the weights of all the embedding layers.
This method takes no extra arguments and unfreezes the sparse weights of the model.

Implementation Ⅱ: unfreeze the weights of a specific embedding layer.

**Arguments**
* `embedding_name`: String, the name of the embedding layer.

***

#### reset_learning_rate_scheduler method

```python
hugectr.Model.reset_learning_rate_scheduler()
```

This method resets the learning rate scheduler of the model. Users can use this method when they want to fine-tune the model weights.

**Arguments**
* `base_lr`: The base learning rate for the internal learning rate scheduler within Model instance. There is NO default value and it should be specified by users.

* `warmup_steps`: The warmup steps for the internal learning rate scheduler within Model instance. The default value is 1.

* `decay_start`: The step at which the learning rate decay starts for the internal learning rate scheduler within Model instance. The default value is 0.

* `decay_steps`: The number of steps of the learning rate decay for the internal learning rate scheduler within Model instance. The default value is 1.

* `decay_power`: The power of the learning rate decay for the internal learning rate scheduler within Model instance. The default value is 2.

* `end_lr`: The final learning rate for the internal learning rate scheduler within Model instance. The default value is 0.

***

#### set_source method

```python
hugectr.Model.set_source()
```

The `set_source` method can set the data source and keyset files under epoch mode training. This overloaded method has two implementations.

Implementation Ⅰ: only valid when `repeat_dataset` is `False` and `use_embedding_training_cache` is `True`.

**Arguments**
* `source`: List[str], the training dataset source. It can be specified with several file lists, e.g., `source = ["file_list.1.txt", "file_list.2.txt"]`. There is NO default value and it should be specified by users.
* `keyset`: List[str], the keyset files. It should be corresponding to the `source`. For example, we can specify `source = ["file_list.1.txt", "file_list.2.txt"]` and `source = ["file_list.1.keyset", "file_list.2.keyset"]`, which have a one-to-one correspondence. There is NO default value and it should be specified by users.
* `eval_source`: String, the evaluation dataset source. There is NO default value and it should be specified by users.

Implementation Ⅱ: only valid when `repeat_dataset` is `False` and `use_embedding_training_cache` is `False`.

**Arguments**
* `source`: String, the training dataset source. For Norm or Parquet dataset, it should be the file list of training data. For Raw dataset, it should be a single training file. There is NO default value and it should be specified by users.
* `eval_source`: String, the evaluation dataset source. For Norm or Parquet dataset, it should be the file list of evaluation data. For Raw dataset, it should be a single evaluation file. There is NO default value and it should be specified by users.

## Low-level Training API

For HugeCTR low-level training API, the core data structures are basically the same as the high-level training API. On this basis, we expose the internal `LearningRateScheduler`, `DataReader` and `EmbeddingTrainingCache` within the `Model`, and provide some low-level training methods as well.HugeCTR currently supports both epoch mode training and non-epoch mode training for dataset in Norm and Raw formats, and only supports non-epoch mode training for dataset in Parquet format. While introducing the API usage, we will elaborate how to employ these two modes of training.

### LearningRateScheduler

#### get_next method

```python
hugectr.LearningRateScheduler.get_next()
```

This method takes no extra arguments and returns the learning rate to be used for the next iteration.

### DataReader

#### set_source method

```python
hugectr.DataReader32.set_source()
hugectr.DataReader64.set_source()
```

The `set_source` method of DataReader currently supports the dataset in Norm and Raw formats, and should be used in epoch mode training. When the data reader reaches the end of file for the current training data or evaluation data, this method can be used to re-specify the training data file or evaluation data file.

**Arguments**
* `file_name`: The file name of the new training source or evaluation source. For Norm format dataset, it takes the form of `file_list.txt`. For Raw format dataset, it appears as `data.bin`. The default value is `''`, which means that the data reader will reset to the beginning of the current data file.

***

#### is_eof method

```python
hugectr.DataReader32.is_eof()
hugectr.DataReader64.is_eof()
```

This method takes no extra arguments and returns whether the data reader has reached the end of the current source file.

### EmbeddingTraingCache

#### update method

```python
hugectr.EmbeddingTraingCache.update()
```

The `update` method of EmbeddingTraingCache currently supports Norm format datasets. Using this method requires that a series of file lists and the corresponding keyset files are generated at the same time when preprocessing the original data to Norm format. This method gives you the ability to load a subset of an embedding table into the GPU in a coarse grained, on-demand manner during the training stage. Please refer to [HugeCTR Embedding Traing Cache](../hugectr_embedding_training_cache.md) if you want to get detailed information about EmbeddingTraingCache.

**Arguments**
* `keyset_file` or `keyset_file_list`: This method is an overloaded method that can accept str or List[str] as an argument. For the model with multiple embedding tables, if the keyset of each embedding table is not separated when generating the keyset files, then pass in the `keyset_file`. If the keyset of each embedding table has been separated when generating keyset files, you need to pass in the `keyset_file_list`, the size of which should equal to the number of embedding tables.

### Model

#### get_learning_rate_scheduler method

```python
hugectr.Model.get_learning_rate_scheduler()
```

`hugectr.Model.get_learning_rate_scheduler` generates and returns the LearningRateScheduler object of the model instance. When the `SGD` optimizer is adopted for training, the returned object can obtain the dynamically changing learning rate according to the `warmup_steps`, `decay_start` and `decay_steps` configured in the `hugectr.CreateSolver` method.
Refer to [SGD Optimizer and Learning Rate Scheduling](hugectr_core_features.md#sgd-optimizer-and-learning-rate-scheduling)) if you want to get detailed information about LearningRateScheduler.

***

#### get_embedding_training_cache method

```python
hugectr.Model.get_embedding_training_cache()
```

This method takes no extra arguments and returns the EmbeddingTrainingCache object.

***

#### get_data_reader_train method

```python
hugectr.Model.get_data_reader_train()
```

This method takes no extra arguments and returns the DataReader object that reads the training data.

***

#### get_data_reader_eval method

```python
hugectr.Model.get_data_reader_eval()
```

This method takes no extra arguments and returns the DataReader object that reads the evaluation data.

***

#### start_data_reading method

```python
hugectr.Model.start_data_reading()
```

This method takes no extra arguments and should be used if and only if it is under the non-epoch mode training. The method starts the `train_data_reader` and `eval_data_reader` before entering the training loop.

***

#### set_learning_rate method

```python
hugectr.Model.set_learning_rate()
```

This method is used together with the `get_next` method of `LearningRateScheduler` and sets the learning rate for the next training iteration.

**Arguments**
* `lr`: Float, the learning rate to be set。

***

#### train method

```python
hugectr.Model.train()
```

This method takes no extra arguments and executes one iteration of the model weights based on one minibatch of training data.

***

#### get_current_loss method

```python
hugectr.Model.get_current_loss()
```

This method takes no extra arguments and returns the loss value for the current iteration.

***

#### eval method

```python
hugectr.Model.eval()
```

This method takes no arguments and calculates the evaluation metrics based on one minibatch of evaluation data.

***

#### get_eval_metrics method

```python
hugectr.Model.get_eval_metrics()
```

This method takes no extra arguments and returns the average evaluation metrics of several minibatches of evaluation data.

***

#### get_incremental_model method

```python
updated_model = hugectr.Model.get_incremental_model()
```

This method is only supported in [Embedding Training Cache](../hugectr_embedding_training_cache.md) and returns the updated embedding table since the last time calling this method to `updated_model`. Note that `updated_model` only stores the embedding features being touched instead of the whole table.

When training with multi-node, the `updated_model` returned in each node doesn't have duplicated embedding features, and the aggregations of `updated_model` from each node form the complete updated sparse model.

The length of `updated_model` is equal to the number of embedding tables in your model, e.g., `length(updated_model)==2` for the wdl model. Each element in `updated_model` is a pair of NumPy arrays: a 1-D array stores keys in `long long` format, and another 2-D array stores embedding vectors in `float` format, where the leading dimension is the embedding vector size. E.g., `updated_model[0][0]` stores keys, and `updated_model[0][1]` stores the embedding vectors corresponding to keys in `updated_model[0][0]`.

***

#### save_params_to_files method

```python
hugectr.Model.save_params_to_files()
```

This method save the model parameters to files. If Embedding Training Cache is utilized, this method will save sparse weights, dense weights and dense optimizer states. Otherwise, this method will save sparse weights, sparse optimizer states, dense weights and dense optimizer states.

The stored sparse model can be used for both the later training and inference cases. Each sparse model will be dumped as a separate folder that contains two files (`key`, `emb_vector`) for the DistributedSlotEmbedding or three files (`key`, `slot_id`, `emb_vector`) for the LocalizedSlotEmbedding. Details of these files are:
* `key`: The unique keys appeared in the training data. All keys are stored in `long long` format, and HugeCTR will handle the datatype conversion internally for the case when `i64_input_key = False`.
* `slot_id`: The key distribution info internally used by the LocalizedSlotEmbedding.
* `emb_vector`: The embedding vectors corresponding to keys stored in the `key` file.

Note that the key, slot id, and embedding vector are stored in the sparse model in the same sequence, so both the nth slot id in `slot_id` file and the nth embedding vector in the `emb_vector` file are mapped to the nth key in the `key` file.

**Arguments**
* `prefix`: String, the prefix of the saved files for model weights and optimizer states. There is NO default value and it should be specified by users.

* `iter`: Integer, the current number of iterations, which will be the suffix of the saved files for model weights and optimizer states. The default value is 0.

***

#### export_predictions method

```python
hugectr.Model.export_predictions()
```

If you want to export the predictions for specified data, using [predict() in inference API](#predict-method) is recommended. This method will export the last batch of evaluation prediction and label to file. If the file already exists, the evaluation result will be appended to the end of the file. This method will only export `eval_batch_size` evaluation result each time. So it should be used in the following way:

```python
for i in range(train_steps):
  # do train
  ...
  # clean prediction / label result file
  prediction_file_in_current_step = "predictions" + str(i)
  if os.path.exists(prediction_file_in_current_step):
    os.remove(prediction_file_in_current_step)
  label_file_in_current_step = "label" + str(i)
  if os.path.exists(label_file_in_current_step):
    os.remove(label_file_in_current_step)
  # do evaluation and export prediction
  for _ in range(solver.max_eval_batches):
    model.eval()
    model.export_predictions(prediction_file_in_current_step, label_file_in_current_step)
```

**Arguments**
* `output_prediction_file_name`: String, the file to which the evaluation prediction results will be writen. The order of the prediction results are the same as that of the labels, but may be different with the order of the samples in the dataset. There is NO default value and it should be specified by users.

* `output_label_file_name`: String, the file to which the evaluation labels will be writen. The order of the labels are the same as that of the prediction results, but may be different with the order of the samples in the dataset. There is NO default value and it should be specified by users.

## Inference API

For HugeCTR inference API, the core data structures are `InferenceParams` and `InferenceModel`. They are designed and implemented for the purpose of multi-GPU offline inference. Please refer to [HugeCTR Backend](https://github.com/triton-inference-server/hugectr_backend) if online inference with Triton is needed.

Please **NOTE** that Inference API requires a configuration JSON file of the model graph, which can derived from the `Model.graph_to_json()` method. Besides, `model_name` within `CreateSolver` should be specified during training in order to correctly dump the JSON file.

### InferenceParams

#### InferenceParams class

```python
hugectr.inference.InferenceParams()
```

`InferenceParams` specifies the parameters related to the inference. An `InferenceParams` instance is required to initialize the `InferenceModel` instance.

**Arguments**
* `model_name`: String, the name of the model to be used for inference. It should be consistent with `model_name` specified during training. There is NO default value and it should be specified by users.

* `max_batchsize`: Integer, the maximum batchsize for inference. It is the global batch size and should be divisible by the length of `deployed_devices`. There is NO default value and it should be specified by users.

* `hit_rate_threshold`: Float, the hit rate threshold for updating the GPU embedding cache. If the hit rate of looking up GPU embedding cahce during inference is below this threshold, then the GPU embedding cache will be updated. The threshold should be between 0 and 1. There is NO default value and it should be specified by users.

* `dense_model_file`: String, the dense model file to be loaded for inference. There is NO default value and it should be specified by users.

* `sparse_model_files`: List[str], the sparse model files to be loaded for inference. There is NO default value and it should be specified by users.

* `use_gpu_embedding_cache`: Boolean, whether to employ the features of GPU embedding cache. If the value is `True`, the embedding vector look up will go to GPU embedding cache. Otherwise, it will reach out to the CPU parameter server directly. There is NO default value and it should be specified by users.

* `cache_size_percentage`: Float, the percentage of cached embeddings on GPU relative to all the embedding tables on CPU.  There is NO default value and it should be specified by users.

* `i64_input_key`: Boolean, this value should be set to `True` when you need to use I64 input key. There is NO default value and it should be specified by users.

* `use_mixed_precision`: Boolean, whether to enable mixed precision inference. The default value is `False`.

* `use_algorithm_search`: Boolean, whether to use algorithm search for cublasGemmEx within the FullyConnectedLayer. The default value is `True`.

* `use_cuda_graph`: Boolean, whether to enable cuda graph for dense network forward propagation. The default value is `True`.

* `deployed_devices`: List[Integer], the list of device id of GPUs. The offline inference will be executed concurrently on the specified multiple GPUs. The default value is `[0]`.

### VolatileDatabaseParams

We provide various volatile database implementations. Generally speaking, two categories can be distinguished.

* **CPU memory databases** are instanced per machine and only use the locally available RAM memory as backing storage. Hence, you may indvidually vary their configuration parameters per machine.

* **Distributed CPU memory databases** are typically shared by all machines in your HugeCTR deployment. They allow you to take advantage of the combined memory capacity of your cluster machines.The configuration parameters for this kind of database should, thus, be identical across all achines in your deployment.

#### VolatileDatabaseParams class

```python
params = hugectr.inference.VolatileDatabaseParams()
params.type = hugectr.DatabaseType_t.<enum_value>
```

Where `<enum_value>` is either:

* `disabled`: Do not use this kind of database.
* `hash_map`: Hash-map based CPU memory database implementation.
* `parallel_hash_map`: Hash-map based CPU memory database implementation with multi threading support **(default)**.
* `redis_cluster`: Connect to an existing Redis cluster deployment (Distributed CPU memory database implementation).


**Configuration of normal hash-map backend**

```python
params.type = hugectr.DatabaseType_t.hash_map
params.algorithm = hugectr.DatabaseHashMapAlgorithm_t.<enum_value>
```

**Configuration of parallelized hash-map backend**

```python
params.type = hugectr.DatabaseType_t.parallel_hash_map
params.algorithm = hugectr.DatabaseHashMapAlgorithm_t.<enum_value>
params.num_partitions = <integer_value>
```

**Configuration of Redis cluster backend**

```python
params.type = "redis_cluster"
params.address = "<host_name_or_ip_address:port_number>"
params.user_name = "<login_user_name>"
params.password = "<login_password>"
params.num_partitions = <int_value>
params.max_get_batch_size = <int_value>
params.max_set_batch_size = <int_value>
```

**Overflow handling related parameters**

To maximize performance and avoid instabilies caused by sporadic high memory usage (*i.e.*, out of memory situations), we provide the overflow handling mechanism. It allows limiting the maximum amount of embeddings to be stored per partition, and, thus, upper-bounding the memory consumption of your distributed database.

```python
params.overflow_margin = <integer_value>
params.overflow_policy = hugectr.DatabaseOverflowPolicy_t.<enum_value>
params.overflow_resolution_target = <double_value>
```

`overflow_margin` denotes the maximum amount of embeddings that will be stored *per partition*. Inserting more than `overflow_margin` embeddings into the database will trigger the execution of the configured `overflow_policy`. Hence, `overflow_margin` upper-bounds the maximum amount of memory that your CPU memory database may occupy. Thumb rule: Larger `overflow_margin` will result higher hit rates, but also increased memory consumption. By **default**, the value of `overflow_margin` is set to `2^64 - 1` (*i.e.*, de-facto infinite). When using the CPU memory database in conjunction with a Persistent database, the idea value for `overflow_margin` may vary. In practice, a setting value to somewhere between `[1 million, 100 million]` tends deliver reliable performance and throughput.

Currently the following values for `overflow_policy` are supported:
* `evict_oldest` **(default)**: Prune embeddings starting from the oldest (i.e., least recently used) until the paratition contains at most `overflow_margin * overflow_resolution_target` embeddings.
* `evict_random`: Prune embeddings random embeddings until the paratition contains at most `overflow_margin * overflow_resolution_target` embeddings.

Unlike `evict_oldest`,  `evict_random` requires no comparison of time-stamps, and thus can be faster. However, `evict_oldest` is likely to deliver better performance over time because embeddings are evicted based on the frequency of their usage. For all eviction policies, `overflow_resolution_target` is expected to be in `]0, 1[` (*i.e.*, between `0` and `1`, but not exactly `0` or `1`). The default value of `overflow_resolution_target` is `0.8` (*i.e.*, the partition is shrunk to 80% of its maximum size, or in other words, when the partition size surpasses `overflow_margin` embeddings, 20% of the embeddings are evicted according to the respective `overflow_policy`).

**Initial caching**

```python
params.initial_cache_rate = <double_value>
```

This is the fraction (`[0.0, 1.0]`) of your dataset that we will attempt to cache immediately upon startup of the parameter server. Hence, setting a value of `0.5` causes the HugeCTR parameter server to attempt caching up to 50% of your dataset directly using the respectively configured volatile database after initialization.

**Refreshing timestamps**

```python
params.refresh_time_after_fetch = <True|False>
```

Some algorithms to organize certain processes, such as the evication of embeddings upon overflow, take time into account. To evalute the affected embeddings, HugeCTR records the time when an embeddings is overridden. This is sufficient in training mode where embeddings are frequently replaced. Hence, the **default value** for this setting is is `false`. However, if you deploy HugeCTR only for inference (*e.g.*, with Triton), this might lead to suboptimal eviction patterns. By setting this value to `true`, HugeCTR will replace the time stored alongside an embedding right after this embedding is accessed. This operation may happen asynchronously (*i.e.*, with some delay).

**Real-time updating**

```python
params.update_filters = [ "<filter 0>", "<filter 1>", ... ]
```

**[Behavior will likely change in future versions]** This setting allows you specify a series of filters, in to permit / deny passing certain model updates from Kafka to the CPU memory database backend. Filters take the form of regular expressions. The **default** value of this setting is `[ ".+" ]` (*i.e.*, process updates for all models, irrespective of their name).

Distributed databases are shared by all your HugeCTR nodes. These nodes will collaborate to inject updates into the underlying database. The assignment of what nodes update what partition may change at runtime.

### PersistentDatabaseParams

Persistent databases are instanced per machine and use the locally available non-volatile memory as backing storage. Hence, you may indvidually vary their configuration parameters per machine.

#### PersistentDatabaseParams class

```python
params = hugectr.inference.PersistentDatabaseParams()
params.type = hugectr.DatabaseType_t.<enum_value>
```

Where `<enum_value>` is either:

* `disabled`: Do not use this kind of database  **(default)**.
* `rocks_db`: Create or connect to a RocksDB database.

**Configuration of RocksDB database backend**

```python
params.type = hugectr.DatabaseType_t.rocks_db
params.path = "<file_system_path>"
params.num_threads = <int_value>
params.read_only = <boolean_value>
params.max_get_batch_size = <int_value>
params.max_set_batch_size = <int_value>
```

`path` denotes the directory in your file-system where the RocksDB database can be found. If the directory does not contain a RocksDB databse, HugeCTR will create an database for you. Note that this may override files that are currently stored in this database. Hence, make sure that `path` points either to an actual RocksDB database or an empty directy. The **default** path is `/tmp/rocksdb`.

`num_threads` is an optimization parameter. This denotes the amount of threads that the RocksDB driver may use internally. By **default**, this value is set to `16`

If the flag `read_only` is set to `true`, the databse will be opened in *Read-Only mode*. Naturally, this means that any attempt to update values in this database will fail. Use for inference, if model is static and the database is shared by multiple nodes (for example via NFS). By **default** this flag is set to `false`.

`max_get_batch_size` and `max_set_batch_size` represent optimization parameters. Mass lookup and insert requests to RocksDB are chunked into batches. For maximum performance `max_*_batch_size` should be large. However, if the available memory for buffering requests in your endpoints is limited, lowering this value may help. By **default**, both values are set to `10000`. With high-performance hardware setups it is **recommended** to increase these values to `1 million`.

**Real-time updating**

```python
params.update_filters = [ "<filter 0>", "<filter 1>", ... ]
```

**[Behavior will likely change in future versions]** This setting allows you specify a series of filters, in to permit / deny passing certain model updates from Kafka to the CPU memory database backend. Filters take the form of regular expressions. The **default value** of this setting is `[ ".+" ]` (*i.e.*, process updates for all models, irrespective of their name).

### UpdateSourceParams

The real-time update source is the origin for model updates during online retraining. To ensure that all database layers are kept in sync, it is advisable configure all nodes in your HugeCTR deployment identical.

#### UpdateSourceParams class

```python
params = hugectr.UpdateSourceParams()
params.type = hugectr.UpdateSourceType_t.<enum_value>
```

Where `<enum_value>` is either:

* `null`: Do not use this kind of database  **(default)**.
* `kafka_message_queue`: Connect to an axisting Apache Kafka message queue.

**Configuration parameters for Apache Kafka update sources**

```python
params.type = hugectr.UpdateSourceType_t.kafka_message_queue
params.brokers = "host_name[:port][;host_name[:port]...]"
params.poll_timeout_ms = <int_value>
params.max_receive_buffer_size = <int_value>
params.max_batch_size <int_value>
params.failure_backoff_ms = <int_value>
```

In order to connect to a Kafka deployments, you need to fill in at least one host-address (hostname + port number) of a Kafka broker node (`brokers` configuration option in the above listings). The **default** value of `brokers` is `127.0.0.1:9092`.

The remaining parameters control certain properties within the notification chain. In particular, `poll_timeout_ms` denotes the maximum time we will wait for additional updates before dispatching them to the database layers in milliseconds. The **default** value is `500` ms.

If, before this limit has run out, more than `max_receive_buffer_size` embedding updates have been received, we will also dispatch these updates immediately. The **default** receive buffer size is `2000`.

Dispatching of updates is conducted in chunks. The maximum size of these chunks is upper-bounded by `max_batch_size`, which is set to `1000` by default.

In some situations, there might be issues that prevent the successful dispatch of an update to a database. For example, if a Redis node is temporarily unreachable. `failure_backoff_ms` is the delay in milliseconds after which we retry dispatching a set of updates in such an event. The **default** backoff delay is `50` ms.

### InferenceModel

#### InferenceModel class

```bash
hugectr.inference.InferenceModel()
```

`InferenceModel` is a collection of inference sessions deployed on multiple GPUs, which can leverage [Hierarchical Parameter Server](../hugectr_parameter_server.md) and enable concurrent execution. The construction of `InferenceModel` requires a model configuration file and an `InferenceParams` instance.

**Arguments**
* `model_config_path`: String, the inference model configuration file (which can be derived from `Model.graph_to_json`). There is NO default value and it should be specified by users.

* `inference_params`: InferenceParams, the InferenceParams object. There is NO default value and it should be specified by users.
***

#### predict method

```python
hugectr.inference.InferenceModel.predict()
```

The `predict` method of InferenceModel makes predictions based on the dataset of Norm or Parquet format. It will return the 2-D numpy array of the shape `(max_batchsize*num_batches, label_dim)`, whose order is consistent with the sample order in the dataset. If `max_batchsize*num_batches` is greater than the total number of samples in the dataset, it will loop over the dataset. For example, there are totally 40000 samples in the dataset, `max_batchsize` equals 4096, `num_batches` equals 10 and `label_dim` equals 2. The returned array will be of the shape `(40960, 2)`, of which first 40000 rows should be desired results and the last 960 rows correspond to the first 960 samples in the dataset.

**Arguments**
* `num_batches`: Integer, the number of prediction batches.

* `source`: String, the source of prediction dataset. It should be the file list for Norm or Parquet format data.

* `data_reader_type`: `hugectr.DataReaderType_t`, the data reader type. We support `hugectr.DataReaderType_t.Norm` and `hugectr.DataReaderType_t.Parquet` currently.

* `check_type`: `hugectr.Check_t`, the check type for the data source. We support `hugectr.Check_t.Sum` and `hugectr.Check_t.Non` currently.

* `slot_size_array`: List[int], the cardinality array of input features. It should be consistent with that of the sparse input. We requires this argument for Parquet format data. The default value is an empty list, which is suitable for Norm format data.

***

#### evaluate method

```python
hugectr.inference.InferenceModel.evaluate()
```

The `evaluate` method of InferenceModel does evaluations based on the dataset of Norm or Parquet format. It requires that the dataset contains the label field. This method returns the AUC value for the specified evaluation batches.

**Arguments**
* `num_batches`: Integer, the number of evaluation batches.

* `source`: String, the source of evaluation dataset. It should be the file list for Norm or Parquet format data.

* `data_reader_type`: `hugectr.DataReaderType_t`, the data reader type. We support `hugectr.DataReaderType_t.Norm` and `hugectr.DataReaderType_t.Parquet` currently.

* `check_type`: `hugectr.Check_t`, the check type for the data source. We support `hugectr.Check_t.Sum` and `hugectr.Check_t.Non` currently.

* `slot_size_array`: List[int], the cardinality array of input features. It should be consistent with that of the sparse input. We requires this argument for Parquet format data. The default value is an empty list, which is suitable for Norm format data.

## Data Generator API

For HugeCTR data generator API, the core data structures are `DataGeneratorParams` and `DataGenerator`. Please refer to `data_generator` directory in the HugeCTR [repository](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/tools) on GitHub to acknowledge how to write Python scripts to generate synthetic dataset and start training HugeCTR model.

### DataGeneratorParams class

```python
hugectr.tools.DataGeneratorParams()
```

`DataGeneratorParams` specifies the parameters related to the data generation. An `DataGeneratorParams` instance is required to initialize the `DataGenerator` instance.

**Arguments**
* `format`: The format for synthetic dataset. The supported types include `hugectr.DataReaderType_t.Norm`, `hugectr.DataReaderType_t.Parquet` and `hugectr.DataReaderType_t.Raw`. There is NO default value and it should be specified by users.

* `label_dim`: Integer, the label dimension for synthetic dataset. There is NO default value and it should be specified by users.

* `dense_dim`:  Integer, the number of dense (or continuous) features for synthetic dataset. There is NO default value and it should be specified by users.

* `num_slot`: Integer, the number of sparse feature slots for synthetic dataset. There is NO default value and it should be specified by users.

* `i64_input_key`: Boolean, whether to use I64 for input keys for synthetic dataset. If your dataset format is Norm or Paruqet, you can choose the data type of each input key. For the Raw dataset format, only I32 is allowed. There is NO default value and it should be specified by users.

* `source`: String, the synthetic training dataset source. For Norm or Parquet dataset, it should be the file list of training data, e.g., source = "file_list.txt". For Raw dataset, it should be a single training file, e.g., source = "train_data.bin". There is NO default value and it should be specified by users.

* `eval_source`: String, the synthetic evaluation dataset source. For Norm or Parquet dataset, it should be the file list of evaluation data, e.g., source = "file_list_test.txt". For Raw dataset, it should be a single evaluation file, e.g., source = "test_data.bin". There is NO default value and it should be specified by users.

* `slot_size_array`: List[int], the cardinality array of input features for synthetic dataset. The list length should be equal to `num_slot`. There is NO default value and it should be specified by users.

* `nnz_array`: List[int], the number of non-zero entries in each slot for synthetic dataset. The list length should be equal to `num_slot`. This argument helps to simulate one-hot or multi-hot encodings. The default value is an empty list and one-hot encoding will be employed then.

* `check_type`: The data error detection mechanism. The supported types include `hugectr.Check_t.Sum` (CheckSum) and `hugectr.Check_t.Non` (no detection). The default value is `hugectr.Check_t.Sum`.

* `dist_type`: The distribution of the sparse input keys for synthetic dataset. The supported types include `hugectr.Distribution_t.PowerLaw` and `hugectr.Distribution_t.Uniform`. The default value is `hugectr.Distribution_t.PowerLaw`.

* `power_law_type`: The specific distribution of power law distribution. The supported types include `hugectr.PowerLaw_t.Long` (alpha=0.9), `hugectr.PowerLaw_t.Medium` (alpha=1.1), `hugectr.PowerLaw_t.Short` (alpha=1.3) and `hugectr.PowerLaw_t.Specific` (requiring a specific alpha value). This argument is only valid when `dist_type` is `hugectr.Distribution_t.PowerLaw`. The default value is `hugectr.PowerLaw_t.Specific`.

* `alpha`: Float, the alpha value for power law distribution. This argument is only valid when `dist_type` is `hugectr.Distribution_t.PowerLaw` and `power_law_type` is `hugectr.PowerLaw_t.Specific`. The alpha value should be greater than zero and not equal to 1.0. The default value is 1.2.

* `num_files`: Integer, the number of training data files that will be generated. This argument is valid when `format` is `hugectr.DataReaderType_t.Norm` or `hugectr.DataReaderType_t.Parquet`. The default value is 128.

* `eval_num_files`: Integer, the number of evaluation data files that will be generated. This argument is valid when `format` is `hugectr.DataReaderType_t.Norm` or `hugectr.DataReaderType_t.Parquet`. The default value is 32.

* `num_samples_per_file`: Integer, the number of samples per generated data file. This argument is valid when `format` is `hugectr.DataReaderType_t.Norm` or `hugectr.DataReaderType_t.Parquet`. The default value is 40960.

* `num_samples`: Integer, the number of samples in the generated single training data file (e.g., train_data.bin). This argument is only valid when `format` is `hugectr.DataReaderType_t.Raw`. The default value is 5242880.

* `eval_num_samples`: Integer, the number of samples in the generated single evaluation data file (e.g., test_data.bin). This argument is only valid when `format` is `hugectr.DataReaderType_t.Raw`. The default value is 1310720.

* `float_label_dense`: Boolean, this is only valid when `format` is `hugectr.DataReaderType_t.Raw`. If its value is set to True, the label and dense features for each sample are interpreted as float values. Otherwise, they are regarded as integer values while the dense features are preprocessed with log(dense[i] + 1.f). The default value is False.

### DataGenerator

#### DataGenerator class

```python
hugectr.tools.DataGenerator()
```

`DataGenerator` provides an API to generate synthetic Norm, Parquet or Raw dataset. The construction of `DataGenerator` requires a `DataGeneratorParams` instance.

**Arguments**
* `data_generator_params`: The DataGeneratorParams instance which encapsulates the required parameters for data generation. There is NO default value and it should be specified by users.

#### generate method

```python
hugectr.tools.DataGenerator.generate()
```

This method takes no extra arguments and starts to generate the synthetic dataset based on the configurations within `data_generator_params`.

## Data Source API

The data source API is used to specify which file system to use in the following training process. There are two data structures: `DataSourceParams` and `DataSource`. Please refer to `data_source` directory in the HugeCTR[repository](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/tools) on GitHub to check out how to write Python scripts to move data from HDFS to local FS.

### DataSourceParams class

```python
hugectr.data.DataSourceParams()
```

`DataSourceParams` specifies the file system information and the paths to data and model used for training. An `DataSourceParams` instance is required to initialize the `DataSource` instance.

**Arguments**
* `use_hdfs`: Boolean, whether to use HDFS or not for dump models. Default is false (use local file system).

* `namenode`: String, the IP address of Hadoop Namenode. Will be ignored if use_hdfs is false. Default is 'localhost'.

* `port`:  Integer, the port of Hadoop Namenode. Will be ignored if use_hdfs is false. Default is 9000.

* `hdfs_train_source`: String, the HDFS path to data used for training.

* `hdfs_train_filelist`: String, the HDFS path to filelist.txt used for training.

* `hdfs_eval_source`: String, the HDFS path to data used to validation.

* `hdfs_eval_filelist`: String, the HDFS path to filelist.txt used for validation.

* `hdfs_dense_model`: String, the HDFS path to load dense model.

* `hdfs_dense_opt_states`: String, the HDFS path to load dense optimizer states.

* `hdfs_sparse_model`: List of strings, the HDFS paths to load sparse models.

* `hdfs_sparse_opt_states`: List of strings, the HDFS paths to load sparse optimizer states.

* `hdfs_model_home`: String, the path to HDFS directory used to store the dumped models and optimizer states.

* `local_train_source`: String, the local path to data used for training.

* `local_train_filelist`: String, the local path to filelist.txt used for training.

* `local_eval_source`: String, the local path to data used to validation.

* `local_eval_filelist`: String, the local path to filelist.txt used for validation.

* `local_dense_model`: String, the local path to load dense model.

* `local_dense_opt_states`: String, the local path to load dense optimizer states.

* `local_sparse_model`: List of strings, the local paths to load sparse models.

* `local_sparse_opt_states`: List of strings, the local paths to load sparse optimizer states.

* `local_model_home`: String, the path to local directory used to store the dumped models and optimizer states.

### DataSource

#### DataSource class

```python
hugectr.data.DataSource())
```

`DataSource` provides an API to help user specify the paths to their data and model file. It can also help user transfer data from HDFS to local filesystem. The construction of `DataSource` requires a `DataSourceParams` instance.

**Arguments**
* `data_source_params`: The DataSourceParams instance.

#### move_to_local method

```python
hugectr.data.DataSource.move_to_local()
```

This method takes no extra arguments and moves all the data user specified in hdfs path to the corresponding local path.

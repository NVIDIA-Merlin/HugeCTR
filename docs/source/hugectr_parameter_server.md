# Hierarchical Parameter Server Database Backend

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

## Introduction to the HPS Database Backend

The Hierarchical Parameter Server database backend (HPS database backend) allows HugeCTR to use models with huge embedding tables by extending HugeCTRs storage space beyond the constraints of GPU memory through utilizing various memory resources across you cluster. Further, it grants the ability to permanently store embedding tables in a structured manner. For an end-to-end demo on how to use the HPS database backend, please refer to [samples](https://github.com/triton-inference-server/hugectr_backend/tree/main/samples/hierarchical_deployment).

## Background

GPU clusters offer superior compute power, compared to their CPU-only counterparts. However, although modern data-center GPUs by NVIDIA are equipped with increasing amounts of memory, new and more powerful AI algorithms come into existence that require more memory. Recommendation models with their huge embedding tables are spearheading these developments. The HPS database backend allows you to efficiently perform inference with models that rely on embedding tables that vastly exceed the available GPU device storage space.

This is achieved through utilizing other memory resources, available within your clsuter, such as CPU accessible RAM and non-volatile memory. Aside from general advantages of non-volatile memory with respect to retaining stored information, storage devices such as HDDs and SDDs offer orders of magnitude more storage space than DDR memory and HBM (High Bandwidth Memory), at significantly lower cost. However, their throughout is lower and latency is higher than that of DRR and HBM.

The HPS database backend acts as an intermediate layer between your GPU and non-volatile memory to store all embeddings of your model. Thereby, available local RAM and/or RAM resources available across the cluster can be used as a cache to improve response times.

## Architecture

As of version 3.3, the HugeCTR hierarchical parameter server database backend defines 3 storage layers.

1. The **CPU Memory Database** layer utilizes volatile CPU addressable RAM memory to cache embeddings.
This database is created and maintained separately by each machine that runs HugeCTR in your cluster.

2. The **Distributed Database** layer allows utilizing Redis cluster deployments to store and retrieve embeddings in and from the RAM memory that is available in your cluster.
The HugeCTR distributed database layer is designed for compatibility with Redis [persistence features](https://redis.io/topics/persistence) such as RDB and AOF to allow seamless continued operation across device restart.
This kind of database is shared by all nodes that participate in the training or inference of a HugeCTR model.

   **Note**: Many products claim Redis compatibility.
   We cannot guarantee or make any statements regarding the suitability of these with our distributed database layer.
   However, we note that Redis alternatives are likely to be compatible with the Redis cluster distributed database layer as long as they are compatible with [hiredis](https://github.com/redis/hiredis).
   We would love to hear about your experiences.
   Please let us know if you successfully or unsuccessfully deployed such Redis alternatives as storage targets with HugeCTR.

3. The **Persistent Database** layer links HugeCTR with a persistent database.
Each node that has such a persistent storage layer configured retains a separate copy of all embeddings in its locally available non-volatile memory.
This layer is best considered as a compliment to the distributed database to further expand storage capabilities and to provide high availability.
As a result, if your model exceeds even the total RAM capacity of your entire cluster or if&mdash;for whatever reason&mdash;the Redis cluster becomes unavailable, all nodes that are configured with a persistent database are still able to respond to inference requests, though likely with increased latency.

The following table provides an overview of the typical properties for the different parameter database layers and the embedding cache.
We emphasize that this table provides rough guidelines.
Properties for production deployments are often different.

|  | GPU Embedding Cache | CPU Memory Database | Distributed Database (InfiniBand) | Distributed Database (Ethernet) | Persistent Database |
|--|--|--|--|--|--|
| Mean Latency | ns ~ us | us ~ ms | us ~ ms | several ms | ms ~ s
| Capacity (relative) | ++  | +++ | +++++ | +++++ | +++++++ |
| Capacity (range in practice) | 10 GBs ~ few TBs  | 100 GBs ~ several TBs | several TBs | several TBs | up to 100s of TBs |
| Cost / Capacity | ++++ | +++ | ++++ | ++++ | + |
| Volatile | yes | yes | configuration dependent | configuration dependent | no |
| Configuration / maintenance complexity | low | low | high | high | low |

## Training and Iterative Model Updates

Models that are deployed with the HugeCTR HPS database backend allow streaming model parameter updates from external sources through [Apache Kafka](https://kafka.apache.org).
This ability provides zero-downtime online model retraining.

## Execution

### Inference

With respect to embedding lookups from the HugeCTR GPU embedding cache and HPS database backend, the following logic applies:

* Whenever the HugeCTR inference engine receives a batch of model input parameters for inference, the inference engine first determines the associated unique embedding keys and tries to resolve these embeddings using the embedding cache.
* When there is a cache miss, the inference engine then turns to the HPS database backend to determine the embedding representations.
* The HPS database backend queries its configured backends in the following order to fill in the missing embeddings:

  1. Local and remote CPU memory locations.
  2. Persistent storage.

HugeCTR first tries to look up missing embeddings in either the CPU memory database or the distributed database.
If, and only if, there are still missing embedding representations after that, HugeCTR tries the non-volatile memory from the persistent database to find the corresponding embedding representations.
The persistent database contains a copy of all existing embeddings.

### Training

After a training iteration, model updates for updated embeddings are published through Kafka by the HugeCTR training process.
The HPS database backend can be configured to listen automatically to change requests for certain models and then ingest these updates in its various database stages.

### Lookup Optimization

If the volatile memory resources&mdash;the CPU memory database and distributed database&mdash;are not sufficient to retain the entire model, HugeCTR attempts to minimize the average latency for lookup through managing these resources like a cache by using a least recently used (LRU) algorithm.

## Configuration

The HugeCTR HPS database backend and iterative update can be configured using three separate configuration objects.
The `VolatileDatabaseParams` and `PersistentDatabaseParams` objects are used to configure the database backends of each HPS database backend instance.
If you want iterative or online model updating, you must also provide the `UpdateSourceParams` object to link the HPS database backend instance with your Kafka deployment.
These objects are part of the [hugectr.inference](./api/python_interface.md#inference-api) Python package.

If you deploy HugeCTR as a backend for NVIDIA [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), you can also provide these configuration options by extending your Triton deployment's JSON configuration:

```json
{
  "supportlonglong": true,

  // ...
  "volatile_db": {
    // ...
  },
  "persistent_db": {
    // ...
  },
  "update_source": {
    // ...
  },
  // ...
}
```

Set the `supportlonglong` field to `True` when you need to use a 64-bit integer input key.
You must set this field to `true` if you specify `True` for the `i64_input_key` parameter.
The default value is `True`.

The following sections describe the configuration parameters.
Generally speaking, each node in your HugeCTR cluster should deploy the same configuration.
In rare cases, it might make sense to vary some parameters.
The most common reason to vary the configuration by node is for heterogeneous clusters.

### Inference Parameters and Embedding Cache Configuration

#### Inference Params Syntax

```python
params = hugectr.inference.InferenceParams(
  model_name = "string",
  max_batchsize = int,
  hit_rate_threshold = 0.9,
  dense_model_file = "string",
  network_file = "string",
  sparse_model_files = ["string-1", "string-2", ...],
  use_gpu_embedding_cache = True,
  cache_size_percentage = 0.2,
  i64_input_key = <True|False>,
  use_mixed_precision = False,
  scaler = 1.0,
  use_algorithm_search = True,
  use_cuda_graph = True,
  number_of_worker_buffers_in_pool = 2,
  number_of_refresh_buffers_in_pool = 1,
  thread_pool_size = 16,
  cache_refresh_percentage_per_iteration = 0.1,
  deployed_devices = [int-1, int-2, ...],
  default_value_for_each_table = [float-1, float-2, ...],
  volatile_db = <volatile-database-configuration>,
  persistent_db = <persistent-database-configuration>,
  update_source = <update-source-parameters>,
  maxnum_des_feature_per_sample = 26,
  refresh_delay = 0.0,
  refresh_interval = 0.0,
  maxnum_catfeature_query_per_table_per_sample = [int-1, int-2, ...],
  embedding_vecsize_per_table = [int-1, int-2, ...],
  embedding_table_names = ["string-1", "string-2", ...]
)
```

The `InferenceParams` object specifies the parameters related to the inference.
An `InferenceParams` object is required to initialize the `InferenceModel` instance.

#### Inference Parameters

* `model_name`: String, specifies the name of the model to use for inference.
It should be consistent with the `model_name` that you specified during training.
This parameter has no default value and you must specify a value.

* `max_batchsize`: Integer, the maximum batch size for inference.
The specific value is the global batch size and should be divisible by the length of `deployed_devices`.
This parameter has no default value and you must specify a value.

* `hit_rate_threshold`: Float, the real hit rate of GPU embedding cache during inference.
When the real hit rate of the GPU embedding cache is higher than the specified threshold, the GPU embedding cache performs an asynchronous insertion of missing embedding keys.
Otherwise, the GPU embedding cache inserts the keys synchronously.
Specify a value between 0 and 1.
The default value is `0.9`

* `dense_model_file`: String, the dense model file to load for inference.
This parameter has no default value and you must specify a value.

* `network_file`: String, specifies a file that includes the model network structure in JSON format.
This file is exported after model training and is used for the initialization of the network structure of the dense part of the model.
This parameter has no default value and you must specify a value.

* `sparse_model_files`: List[str], the sparse model files to load for inference.
This parameter has no default value and you must specify a value.

* `device_id`: Integer, is scheduled to be deprecated and replaced by `devicelist`.

* `use_gpu_embedding_cache`: Boolean, whether to employ the features of GPU embedding cache.
When set to `True`, the embedding vector look up goes to the GPU embedding cache.
Otherwise, the look up attempts to use the CPU HPS database backend directly.
The default value is `True`.

* `cache_size_percentage`: Float, the percentage of cached embeddings on the GPU, relative to all the embedding tables on the CPU.
The default value is `0.2`.

* `i64_input_key`: Boolean, this value should be set to `True` when you need to use an Int64 input key.
This parameter has no default value and you must specify a value.

* `use_mixed_precision`: Boolean, whether to enable mixed precision inference.
The default value is `False`.

* `scaler`: Float, the scaler to use when mixed precision training is enabled.
The function supports `128`, `256`, `512`, and `1024` scalers only for mixed precision training.
The default value is `1.0` and corresponds to no mixed precision training.

* `use_algorithm_search`: Boolean, whether to use algorithm search for `cublasGemmEx` within the fully connected layer.
The default value is `True`.

* `use_cuda_graph`: Boolean, whether to enable CUDA graph for dense-network forward propagation.
The default value is `True`.

* `number_of_worker_buffers_in_pool`: Integer, specifies the number of worker buffers to allocate in the embedded cache memory pool.
Specify a value such as two times the number of model instances to avoid resource exhaustion.
An alternative to specifying a larger value while still avoiding resource exhaustion is to disable asynchronous updates by setting the `hit_rate_threshold` parameter to greater than `1`.
The default value is `2`.

* `number_of_refresh_buffers_in_pool`: Integer, specifies the number of refresh buffers to allocate in the embedded cache memory pool.
HPS uses the refresh memory pool to support online updates of incremental models.
Specify larger values if model updates occur at a high-frequency or you have a large volume of incremental model updates.
The default value is `1`.

* `thread_pool_size`: Integer, specifies the size of the thread pool. The thread pool is used by the GPU embedding cache to perform asynchronous insertion of missing keys.
The actual thread pool size is set to the maximum of the value that you specify and the value returned by `std::thread::hardware_concurrency()`.
The default value is `16`.

The actual thread pool size will be set as the maximum value of this configured one and `std::thread::hardware_concurrency()`.
The default value is `16`.

* `cache_refresh_percentage_per_iteration`: Float, specifies the percentage of the embedding cache to refresh during each iteration.
To avoid reducing the performance of the GPU cache during online updating, you can configure the update percentage of GPU embedding cache.
For example, if you specify `cache_refresh_percentage_per_iteration = 0.2`, the entire GPU embedding cache is refreshed during 5 iterations.
Specify a smaller value if model updates occur at a high-frequency or you have a large volume of incremental model updates.
The default value is `0.1`.

* `deployed_devices`: List[Integer], specifies a list of the device IDs of your GPUs.
The offline inference is executed concurrently on the specified GPUs.
The default value is `[0]`.

* `default_value_for_each_table`:List[Float], specifies a default value when an embedding key cannot be returned.
When an embedding key can not be queried in the GPU cache or volatile and persistent databases, the default value is returned.
For models with multiple embedding tables, each embedding table has a default value.

* `volatile_db`: See the [Volatile Database Configuration](#volatile-database-configuration) section.

* `persistent_db`: See the [Persistent Database Configuration](#persistent-database-configuration) section.

* `update_source`: See the [Update Source Configuration](#update-source-configuration) section.

* `maxnum_des_feature_per_sample`: Integer, specifies the maximum number of dense features in each sample.
Because each sample can contain a varying number of numeric (dense) features, use this parameter to specify the maximum number of dense feature in each sample.
The specified value determines the pre-allocated memory size on the host and device.
The default value is `26`.

* `refresh_delay`: Float, specifies an initial delay, in seconds, to wait before beginning to refresh the embedding cache.
The timer begins when the service launches.
The default value is `0.0`.

* `refresh_interval`: Float, specifies the interval, in seconds, for the periodic refresh of the embedding keys in the GPU embedding cache.
The embedding keys are refreshed from volatile and persistent data sources based on the specified number of seconds.
The default value is `0.0`.

* `maxnum_catfeature_query_per_table_per_sample`: List[Int], this parameter determines the pre-allocated memory size on the host and device.
We assume that for each input sample, there is a maximum number of embedding keys per sample in each embedding table that need to be looked up.
Specify this parameter as [max(the number of embedding keys that need to be queried from embedding table 1 in each sample), max(the number of embedding keys that need to be queried from embedding table 2 in each sample), ...]
This parameter has no default value and you must specify a value.

* `embedding_vecsize_per_table`:List[Int], this parameter determines the pre-allocated memory size on the host and device.
For the case of multiple embedding tables, we assume that the size of the embedding vector in each embedding table is different.
Specify the maximum vector size for each embedding table.
This parameter has no default value and you must specify a value.

* `embedding_table_names`: List[String], specifies the name of each embedding table.
The names are used to name the data partition and data table in the hierarchical database backend.
The default value is `["sparse_embedding1", "sparse_embedding2", ...]`

* `label_dim`: Int, each model can contain a varying size of prediction result, such as a multi-task model.
Specify the maximum size of prediction result in each sample.
The specified value determines the pre-allocated memory size on the host and device.
The default value is `1`.

* `slot_num`: Int, each model can contain a fixed size of feature fields.
Specify the number of feature fields (the number of slots).
The specified value determines the pre-allocated memory size on the host and device.
The default value is `10`.


#### Parameter Server Configuration: Models

The following JSON shows a sample configuration for the `models` key in a parameter server configuration file.

```json
"supportlonglong": true,
"models":[
  {
    "model":"wdl",
    "sparse_files":["/wdl_infer/model/wdl/1/wdl0_sparse_20000.model", "/wdl_infer/model/wdl/1/wdl1_sparse_20000.model"],
    "dense_file":"/wdl_infer/model/wdl/1/wdl_dense_20000.model",
    "network_file":"/wdl_infer/model/wdl/1/wdl.json",
    "num_of_worker_buffer_in_pool": 4,
    "num_of_refresher_buffer_in_pool": 1,
    "deployed_device_list":[0],
    "max_batch_size":64,
    "default_value_for_each_table":[0.0,0.0],
    "maxnum_des_feature_per_sample":26,
    "maxnum_catfeature_query_per_table_per_sample":[2,26],
    "embedding_vecsize_per_table":[1,15],
    "embedding_table_names":["table1","table2"],
    "refresh_delay":0,
    "refresh_interval":0,
    "hit_rate_threshold":0.9,
    "gpucacheper":0.1,
    "gpucache":true,
    "cache_refresh_percentage_per_iteration": 0.2,
    "label_dim": 1,
    "slot_num":10
  }
]
```

### Volatile Database Configuration

For HugeCTR, the volatile database implementations are grouped into two categories:

* **CPU memory databases** have an instance on each machine and only use the locally available RAM memory as backing storage.
As a result, you can indvidually vary their configuration parameters for each machine.

* **Distributed CPU memory databases** are typically shared by all machines in your HugeCTR deployment.
They enable you to use the combined memory capacity of your cluster machines.
The configuration parameters for this kind of database should be identical across all machines in your deployment.

  Distributed databases are shared by all your HugeCTR nodes.
  These nodes collaborate to inject updates into the underlying database.
  The assignment of which nodes update specific partition can change at runtime.

#### Volatile Database Params Syntax

```python
params = hugectr.inference.VolatileDatabaseParams(
  type = "redis_cluster",
  address = "127.0.0.1:7000",
  user_name = "default",
  password = "",
  num_partitions = int,
  allocation_rate = 268435456,
  max_get_batch_size = 10000,
  max_set_batch_size = 10000,
  overflow_margin = int,
  overflow_policy = hugectr.DatabaseOverflowPolicy_t.<enum_value>,
  overflow_resolution_target = 0.8,
  initial_cache_rate = 1.0,
  refresh_time_after_fetch = False,
  cache_missed_embeddings = False,
  update_filters = ["filter-0", "filter-1", ...]
)
```

#### Parameter Server Configuration: Volatile Database

The following JSON shows a sample configuration for the `volatile_db` key in a parameter server configuration file.

```json
"volatile_db": {
  "type": "redis_cluster",
  "address": "127.0.0.1:7003,127.0.0.1:7004,127.0.0.1:7005",
  "user_name":  "default",
  "password": "",
  "num_partitions": 8,
  "allocation_rate": 268435456,
  "max_get_batch_size": 10000,
  "max_set_batch_size": 10000,
  "overflow_margin": 10000000,
  "overflow_policy": "evict_oldest",
  "overflow_resolution_target": 0.8,
  "initial_cache_rate": 1.0,
  "cache_missed_embeddings": false,
  "update_filters": [".+"]
}
```

#### Volatile Database Parameters

* `type`: specifies the volatile database implementation.
Specify one of the following:

  * `hash_map`: Hash-map based CPU memory database implementation.
  * `parallel_hash_map`: Hash-map based CPU memory database implementation with multi threading support. This is the default value.
  * `redis_cluster`: Connect to an existing Redis cluster deployment (Distributed CPU memory database implementation).

The following parameters apply when you set `type="hash_map"` or `type="parallel_hash_map"`:

* `num_partitions`: Integer, specifies the number of partitions for embedding tables and controls the degree of parallelism.
Parallel hashmap implementations split your embedding tables into approximately evenly-sized partitions and parallelizes look up and insert operations.
The default value is calculated as `min(number_of_cpu_cores, 16)` of the system that you used to build the HugeCTR binaries.

* `allocation_rate`: Integer, specifies the maximum number of bytes to allocate for each memory allocation request.
The default value is `268435456` bytes, 256 MiB.

The following parameters apply when you set `type="redis_cluster"`:

* `address`: String, specifies the address of one of servers of the Redis cluster.
Use the pattern `"host-1[:port],host-2[:port],..."`.
The default value is `"127.0.0.1:7000"`.

* `user_name`: String, specifies the user name of the Redis cluster.
The default value is `"default"`.

* `password`: String, specifies the password of your account.
The default value is `""` and corresponds to no password.

* `num_partitions`: Integer, specifies the number of partitions for embedding tables.
Each embedding table is divided into `num_partitions` of approximately evenly-sized partitions.
Each partition is assigned a storage location in your Redis cluster.
HugeCTR does not provide any guarantees regarding the placement of partitions.
As a result, multiple partitions can be stored the same node for some models and deployments.
In most cases, to take advantage of your cluster resources, set `num_partitions` to at least equal to the number of Redis nodes.
For optimal performance, set `num_parititions` to be strictly larger than the number of machines.
However, each partition incurs a small processing overhead so do not specify a value that is too large.
A typical value that retains high performance and provides good cluster utilization is 2-5x the number of machines in your Redis cluster.
The default value is `8`.

* `max_get_batch_size` and `max_set_batch_size`: Integer, specifies optimization parameters.
Mass lookup and insert requests to distributed endpoints are chunked into batches.
For maximum performance, these two parameters should be large.
However, if the available memory for buffering requests in your endpoints is limited or you experience transmission stability issues, specifying smaller values can help.
By default, both parameters are set to `10000`.
With high-performance networking and endpoint hardware, try setting the values to `1000000`.

#### Overflow Parameters

To maximize performance and avoid instabilities that can be caused by sporadic high memory usage, such as an out of memory situations, HugeCTR provides an overflow handling mechanism.
This mechanism enables you to limit the maximum amount of embeddings to store for each partition.
The limit acts as an upper bound for the memory consumption of your distributed database.

* `overflow_margin`: Integer, specifies the maximum amount of embeddings to store for each partition.
Inserting more than `overflow_margin` embeddings into the database triggers the  configured `overflow_policy`.
This parameter sets the upper bound for the maximum amount of memory that your CPU memory database can occupy.
Larger values for this parameter result in higher hit rates but also consume more memory.
The default value is `2^64 - 1` and indicates no limit.

  When you use a CPU memory database in conjunction with a persistent database, the ideal value for `overflow_margin` can vary.
  In practice, a value in the range `[1000000, 100000000]` provides reliable performance and throughput.

* `overflow_policy`: specifies how to respond to an overflow condition.
Specify one of the following:
  * `evict_oldest`: Prune embeddings starting from the oldest embedding
  until the partition contains at most `overflow_margin * overflow_resolution_target` embeddings. This policy implements the least-recently used (LRU) algorithm.
  * `evict_random`: Prune embeddings at random until the partition contains at most `overflow_margin * overflow_resolution_target` embeddings.

  Unlike `evict_oldest`, the `evict_random` policy does not require a comparison of timestamps and can be faster.
  However, `evict_oldest` is likely to deliver better performance over time because the policy evicts embeddings based on the frequency of their use.

* `overflow_resolution_target`: Double, specifies the fraction of the embeddings to keep when embeddings must be evicted.
Specify a value between `0` and `1`, but not exactly `0` or `1`.
The default value is `0.8` and indicates to evict embeddings from a partition until it is shrunk to 80% of its maximum size.
In other words, when the partition size surpasses `overflow_margin` embeddings, 20% of the embeddings are evicted according to the specified `overflow_policy`.

* `initial_cache_rate`: Double, specifies the fraction of the embeddings to initially attempt to cache.
Specify a value in the range `[0.0, 1.0]`.
HugeCTR attempts to cache the specified fraction of the dataset immediately upon startup of the HPS database backend.
For example, a value of `0.5` causes the HugeCTR HPS database backend to attempt to cache up to 50% of your dataset using the volatile database after initialization.
The default value is `1.0`.

* `refresh_time_after_fetch`: Bool, when set to `True`, the timestamp for an embedding is updated after the embedding is accessed.
The timestamp update can be performed asynchronously and experience some delay.
Some algorithms for overflow and eviction take time into account.
To evaluate the affected embeddings, HugeCTR records the time when an embedding is overwritten.
The default value is `False` and this setting is sufficient when you train a model and embeddings are frequently replaced.
However, when you deploy HugeCTR only for inference, such as with Triton Inference Server, the default setting might lead to suboptimal eviction patterns.

#### Common Volatile Database Parameters

The following parameters are common to all volatile database types.

* `cache_missed_embeddings`: Bool, when set to `True` and an embedding could not be retrieved from the volatile database, but could be retrieved from the persistent database, the embedding is inserted into the volatile database.
The insert operation could replace another value.
The default value is `False` and disables this functionality.

  This setting optimizes the volatile database in response to the queries that are received in inference mode.
  In training mode, updated embeddings are automatically written back to the database after each training step.
  As a result, setting the value to `True` during training is likely to increase the number of writes to the database and degrade performance without providing significant improvements.

* `update_filters`: List[str], specifies regular expressions that are used to control sending model updates from Kafka to the CPU memory database backend.
The default value is `["^hps_.+$"]` and processes updates for all HPS models because the filter matches all HPS model names.

  The functionality of this parameter might change in future versions.


### Persistent Database Configuration

Persistent databases have an instance on each machine and use the locally available non-volatile memory as backing storage.
As a result, some configuration parameters can vary according to the specifications of the machine.

#### Persistent Database Params Syntax

```python
params = hugectr.inference.PersistentDatabaseParams(
  type = hugectr.DatabaseType_t.<enum_value>,
  path = "/tmp/rocksdb",
  num_threads = 16,
  read_only = False,
  max_get_batch_size = 10000,
  max_set_batch_size = 10000,
  update_filters = ["filter-0", "filter-1", ... ]
)
```

#### Parameter Server Configuration: Persistent Database

The following JSON shows a sample configuration for the `persistent_db` key in a parameter server configuration file.

```json
"persistent_db": {
  "type": "rocks_db",
  "path": "/tmp/rocksdb",
  "num_threads": 16,
  "read_only": false,
  "max_get_batch_size": 10000,
  "max_set_batch_size": 10000,
  "update_filters": [".+"]
}
```

#### Persistent Database Parameters

* `type`: specifies the persistent datatabase implementation.
Specify one of the following:
  * `disabled`: Prevents the use of a persistent database.
  This is the default value.
  * `rocks_db`: Create or connect to a RocksDB database.

* `path` String, specifies the directory on each machine where the RocksDB database can be found.
If the directory does not contain a RocksDB database, HugeCTR creates a database for you.
Be aware that this behavior can overwrite files that are stored in the directory.
For best results, make sure that `path` specifies an existing RocksDB database or an empty directory.
The default value is `/tmp/rocksdb`.

* `num_threads` Int, specifies the number of threads for the RocksDB driver.
The default value is `16`.

* `read_only`, Bool, when set to `True`, the database is opened in read-only mode.
Read-only mode is suitable for use with inference if the model is static and the database is shared by multiple machines, such as with NFS.
The default value is `False`.

* `max_get_batch_size` and `max_set_batch_size`, Int, specifies the batch size for lookup and insert requests.
Mass lookup and insert requests to RocksDB are chunked into batches.
For maximum performance these parameters should be large.
However, if the available memory for buffering requests in your endpoints is limited, lowering this value might improve performance.
The default value for both parameters is `10000`.
With high-performance hardware, you can attempt to set these parameters to `1000000`.

* `update_filters`: List[str], specifies regular expressions that are used to control sending model updates from Kafka to the CPU memory database backend.
The default value is `["^hps_.+$"]` and processes updates for all HPS models because the filter matches all HPS model names.

  The functionality of this parameter might change in future versions.

### Update Source Configuration

The real-time update source is the origin for model updates during online retraining.
To ensure that all database layers are kept in sync, configure all the nodes in your HugeCTR deployment identically.

#### Update Source Params Syntax

```python
params = hugectr.UpdateSourceParams(
  type = "kafka_message_queue",
  brokers = "host-1[:port][;host-2[:port]...]",
  metadata_refresh_interval_ms = 30000,
  poll_timeout_ms = 500,
  receive_buffer_size = 262144,
  max_batch_size = 8192,
  failure_backoff_ms = 50
  max_commit_interval = 32
)
```

#### Parameter Server Configuration: Update Source

The following JSON shows a sample configuration for the `update_source` key in a parameter server configuration file.

```json
"update_source": {
  "type": "kafka_message_queue",
  "brokers": "127.0.0.1:9092",
  "metadata_refresh_interval_ms": 30000,
  "poll_timeout_ms": 500,
  "receive_buffer_size": 262144,
  "max_batch_size": 8192,
  "failure_backoff_ms": 50,
  "max_commit_interval": 32
}
```

#### Update Source Parameters

* `type`: String, specifies the update source implementation.
Specify one of the following:
  * `null`: Prevents the use of an update source. This is the default value.
  * `kafka_message_queue`: Connect to an existing Apache Kafka message queue.

* `brokers`: String, specifies a semicolon-delimited list of host name or IP address and port pairs.
You must specify  at least one host name and port of a Kafka broker node.
The default value is `127.0.0.1:9092`.

* `metadata_refresh_interval_ms`: Int, specifies the frequency at which the topic metadata downloaded from the Kafka broker.

* `receive_buffer_size` Int, specifies the size of the buffer, in bytes, that stores data that is received from Kafka.
The best value to specify is equal to `send_buffer_size` of the KafkaMessageSink that is used to push updates to Kafka.
The `message.max.bytes` setting of the Kafka broker must be at least `receive_buffer_size + 1024` bytes.
The default value is `262144` bytes.

* `poll_timeout_ms`: Int, specifies the maximum time to wait, in milliseconds, for additional updates before dispatching updates to the database layers.
The default value is `500` ms.

* `max_batch_size`: Int, specifies the maximum number of keys and values from messages to consume before dispatching updates to the database.
HugeCTR dispatches the updates in chunks.
The maximum size of these chunks is set with this parameter.
The default value is `8192`.

* `failure_backoff_ms`: Int, specifies a delay, in milliseconds, to wait after failing to dispatch updates to the database successfully.
In some situations, there can be issues that prevent the successful dispatch such as if a Redis node is temporarily unreachable.
After the delay, HugeCTR retries dispatching a set of updates.
The default value is `50` ms.

* `max_commit_interval`: Int, specifies the maximum number of messages to hold before delivering and committing the messages to Kafka.
This parameter is evaluated independent of any other conditions or parameters.
Any received data is forwarded and committed if at most `max_commit_interval` were processed since the previous commit.
The default value is `32`.

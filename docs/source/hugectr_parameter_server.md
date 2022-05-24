# HugeCTR Hierarchical Parameter Server Database Backend


## Introduction

The hierarchical parameter server database backend (HPS database backend) allows HugeCTR to use models with huge embedding tables by extending HugeCTRs storage space beyond the constraints of GPU memory through utilizing various memory resources across you cluster. Further, it grants the ability to permanently store embedding tables in a structured manner. For an end-to-end demo on how to use the HPS database backend, please refer to [samples](https://github.com/triton-inference-server/hugectr_backend/tree/main/samples/hierarchical_deployment).


## Background

GPU clusters offer superior compute power, compared to their CPU-only counterparts. However, although modern data-center GPUs by NVIDIA are equipped with increasing amounts of memory, new and more powerful AI algorithms come into existence that require more memory. Recommendation models with their huge embedding tables are spearheading these developments. The HPS database backend allows you to efficiently perform inference with models that rely on embedding tables that vastly exceed the available GPU device storage space.

This is achieved through utilizing other memory resources, available within your clsuter, such as CPU accessible RAM and non-volatile memory. Aside from general advantages of non-volatile memory with respect to retaining stored information, storage devices such as HDDs and SDDs offer orders of magnitude more storage space than DDR memory and HBM (High Bandwidth Memory), at significantly lower cost. However, their throughout is lower and latency is higher than that of DRR and HBM.

The HPS database backend acts as an intermediate layer between your GPU and non-volatile memory to store all embeddings of your model. Thereby, available local RAM and/or RAM resources available across the cluster can be used as a cache to improve response times.


## Architecture

As of version 3.3, the HugeCTR hierarchical parameter server database backend defines 3 storage layers.

1. 
   The **CPU Memory Database** layer
   utilizes volatile CPU addressable RAM memory to cache embeddings. This database is created and maintained separately by each machine that runs HugeCTR in your cluster.

2. 
   The **Distributed Database** layer allows utilizing Redis cluster deployments, to store and retrieve embeddings in/from the RAM memory available in your cluster. The HugeCTR distributed database layer is designed for compatibility with Redis [peristence features](https://redis.io/topics/persistence) such as [RDB](https://redis.io/topics/persistence) and [AOF](https://redis.io/topics/persistence) to allow seamless continued operation across device restart. This kind of databse is shared by all nodes that participate in the training / inference of a HugeCTR model.
   
   *Remark: There exists and abundance of products that claim Redis compatibility. We cannot guarantee or make any statements regarding the suitabablity of these with our distributed database layer. However, we note that Redis alternatives are likely to be compatible with the Redis cluster dstributed database layer, as long as they are compatible with [hiredis](https://github.com/redis/hiredis). We would love to hear about your experiences. Please let us know if you successfully/unsuccessfully deployed such Redis alternatives as storage targets with HugeCTR.*

3. 
   The **Persistent Database** layer links HugeCTR with a persistent database. Each node that has such a persistent storage layer configured retains a separate copy of all embeddings in its locally available non-volatile memory. This layer is best considered as a compliment to the distributed database to 1) further expand storage capabilities and 2) for high availability. Hence, if your model exceeds even the total RAM capacity of your entire cluster, or if - for whatever reason - the Redis cluster becomes unavailable, all nodes that have been configured with a persistent database will still be able to fully cater to inference requests, albeit likely with increased latency.

In the following table, we provide an overview of the *typical* properties different parameter database layers (and the embedding cache). We emphasize that this table is just intended to provide a rough orientation. Properties of actual deployments may deviate.


|  | GPU Embedding Cache | CPU Memory Database | Distributed Database (InfiniBand) | Distributed Database (Ethernet) | Persistent Database |
|--|--|--|--|--|--|
| Mean Latency | ns ~ us | us ~ ms | us ~ ms | several ms | ms ~ s
| Capacity (relative) | ++  | +++ | +++++ | +++++ | +++++++ |
| Capacity (range in practice) | 10 GBs ~ few TBs  | 100 GBs ~ several TBs | several TBs | several TBs | up to 100s of TBs |
| Cost / Capacity | ++++ | +++ | ++++ | ++++ | + |
| Volatile | yes | yes | configuration dependent | configuration dependent | no |
| Configuration / maintenance complexity | low | low | high | high | low |


## Training and Iterative Model Updates

Models deployed via the HugeCTR HPS database backend allow streaming model parameter updates from external sources via [Apache Kafka](https://kafka.apache.org). This function allows zero-downtime online model re-training - for example using the HugeCTR model training system.


## Execution

### Inference

With respect to embedding lookups via the HugeCTR GPU embedding cache and HPS database backend, the following logic applies. Whenever the HugeCTR inference engine receives a batch of model input parameters for inference, it will first determine the associated unique embedding keys and try to resolve these embeddings using the embedding cache. If there are cache misses, it will then turn to the HPS database backend to determine the embedding representations. The HPS database backend queries its configured backends in the following order to fill in the missing embeddings:

1. Local / Remote CPU memory locations
2. Permanent storage

Hence, if configured, HugeCTR will first try to lookup missing embeddings in either a *CPU Memory Database* or *Distributed Database*. If and only if, there are still missing embedding representations after that, HugeCTR will turn to non-volatile memory (via the *Persistent Database*, which contains a copy of all existing embeddings) to find the corresponding embedding representations.

### Training

After a training iteration, model updates for updated embeddings are published via Kafka by the HugeCTR training process. The HPS database backend can be configured to automatically listen to change requests for certain models and will ingest these updates in its various database stages.


### Lookup optimization

If volatile memory resources, i.e. the *CPU Memory Database* and/or *Distributed Database*, are not sufficient to retain the entire model, the HugeCTR will attempt to minimize the avarage latency for lookup through managing these resources like cache using an LRU paradigm.


## Configuration

The HugeCTR HPS database backend and iterative update can be configured using 3 separate configuration objects. Namely, the `VolatileDatabaseParams` and `PersistentDatabaseParams` are used to configure the database backends of each HPS database backend instance. If you desire iterative or online model updating, you must also provide the `UpdateSourceParams` configuration object to link the HPS database backend instance with your Kafka reployment. These objects are part of the Python package [hugectr.inference](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html#inference-api).

If you deploy HugeCTR as a backend for [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), you can also provide these configuration options by extending your Triton deployment's JSON configuration:

```text
{
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

Next we will describe all available configuration options. Generally speaking, each node in your HugeCTR cluster should deploy the same configuration. However, it may make sense to vary some arguments in some situations, especially in heterogeneous cluster setups.

### **Inference Parameters and Embedding Cache Configurations**

**Python**
```python
hugectr.inference.InferenceParams()
```
`InferenceParams` specifies the parameters related to the inference. An `InferenceParams` instance is required to initialize the `InferenceModel` instance.

**Arguments**
* `model_name`: String, the name of the model to be used for inference. It should be consistent with `model_name` specified during training. There is NO default value and it should be specified by users.

* `max_batchsize`: Integer, the maximum batchsize for inference. It is the global batch size and should be divisible by the length of `deployed_devices`. There is NO default value and it should be specified by users.

* `hit_rate_threshold`: Float, If the real hit rate of GPU embedding cahce during inference If the real hit rate of GPU embedding cache is lower than this threshold, the GPU cache will perform synchronous insertion of missing embedding keys, otherwise asynchronous insertion. The threshold should be between 0 and 1. The default value is `0.9`

* `dense_model_file`: String, the dense model file to be loaded for inference. There is NO default value and it should be specified by users.

* `sparse_model_files`: List[str], the sparse model files to be loaded for inference. There is NO default value and it should be specified by users.

* `device_id`: Int, is about to be deprecated and replaced by `devicelist`.

* `use_gpu_embedding_cache`: Boolean, whether to employ the features of GPU embedding cache. If the value is `True`, the embedding vector look up will go to GPU embedding cache. Otherwise, it will reach out to the CPU HPS database backend directly. The default value is `true`.

* `cache_size_percentage`: Float, the percentage of cached embeddings on GPU relative to all the embedding tables on CPU.  The default value is `0.2`.

* `i64_input_key`: Boolean, this value should be set to `True` when you need to use I64 input key. There is NO default value and it should be specified by users.

* `use_mixed_precision`: Boolean, whether to enable mixed precision inference. The default value is `False`.

* `scaler`: Float, the scaler to be used when mixed precision training is enabled. Only 128, 256, 512, and 1024 scalers are supported for mixed precision training. The default value is 1.0, which corresponds to no mixed precision training. 

* `use_algorithm_search`: Boolean, whether to use algorithm search for cublasGemmEx within the FullyConnectedLayer. The default value is `True`.

* `use_cuda_graph`: Boolean, whether to enable cuda graph for dense network forward propagation. The default value is `True`.

* `number_of_worker_buffers_in_pool`: Int, since HPS supports asynchronous or synchronous insertion of missing keys by worker memory pool. The size of the worker memory pool is determined by **num_of_worker_buffer_in_pool**. It is recommended to increase the size of the memory pool (such as 2 times the number of model instance) in order to avoid resource exhaustion or disable asynchronous updates (set the `hit_rate_threshold` to greater than 1). The default value is `1`.

* `number_of_refresh_buffers_in_pool`: Int, HPS supports online updates of incremental models by refresh memory pool. The size of the refresh memory pool is determined by **number_of_refresh_buffers_in_pool**. It is recommended to increase the size of the memory pool for high-frequency and large size of incremental model update. The default value is `1`.

* `cache_refresh_percentage_per_iteration`: Float, in order not to affect the performance of the gpu cache during online updating, the user can configure the update percentage of GPU embedding cache. For example, if cache_refresh_percentage_per_iteration=0.2, it means that the entire GPU embedding cache will be refreshed through 5 iterations. It is recommended to use a smaller refresh percentage for high-frequency and large size of incremental model update. The default value is `0.1`.

* `deployed_devices`: List[Integer], the list of device id of GPUs. The offline inference will be executed concurrently on the specified multiple GPUs. The default value is `[0]`.

* `default_value_for_each_table`:List[Float], for the embedding key that cannot be queried in the gpu cache and volatile/persistent database, the default value will be returned directly. For models with multiple embedding tables, each embedding table has a default value.

* `volatile_db`: see the [`Volatile Database Configurations`](#volatile-database-configurations) part below.

* `persistent_db`: see the [`Persistent Database Configurations`](#persistent-database-configurations) part below.

* `update_source`: see the [`Update Source Configurations*`](#update-source-configurations) part below.

**The followings are embedding cache related parameters**:

* `maxnum_des_feature_per_sample`: Int, each sample may contain a varying number of numeric (dense) features. This item so the user needs to configure the value of Maximum(the number of dense feature in each sample) in this item, which determines the pre-allocated memory size on the host and device. The default value is `26`.

* `refresh_delay`: Float, the embedding keys in the GPU embedding cache are once refreshed from a volatile/persistent database after the "refresh_delay" seconds (start timing after the service launches) configured by the user. The default value is `0.0`.

* `refresh_interval`: Float, the embedding keys in the GPU embedding cache are periodically refreshed from volatile/persistent database based on the "refresh_interval" seconds (start timing after the service launches) configured by user. The default value is `0.0`.

* `maxnum_catfeature_query_per_table_per_sample`: List[Int], this item determines the pre-allocated memory size on the host and device. We assume that for each input sample, there is a maximum number of embedding keys per sample in each embedding table that need to be looked up, so the user needs to configure the [ Maximum(the number of embedding keys that need to be queried from embedding table 1 in each sample), Maximum(the number of embedding keys that need to be queried from embedding table 2 in each sample), ...] in this item. This is a mandatory configuration item.

* `embedding_vecsize_per_table`:List[Int], this item determines the pre-allocated memory size on the host and device.  For the case of multiple embedding tables, we assume that the size of the embedding vector in each embedding table is different, then this configuration item requires the user to fill in each embedding table with maximum vector size. This is a mandatory configuration item.

* `embedding_table_names`: List[String], this configuration item needs to be filled with the name of each embedded table, which will be used to name the data partition and data table in the hierarchical database backend. The default value is `["sparse_embedding1", "sparse_embedding2", ...]`

**JSON(ps.json) Example:**
```text
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
            "hit_rate_threshold":0.9,
            "gpucacheper":0.1,
            "gpucache":true,
            "cache_refresh_percentage_per_iteration": 0.2
        }
    ] 
```


### **Volatile Database Configurations**
We provide various volatile database implementations. Generally speaking, two categories can be distinguished.

* **CPU memory databases** are instanced per machine and only use the locally available RAM memory as backing storage. Hence, you may indvidually vary their configuration parameters per machine.

* **Distributed CPU memory databases** are typically shared by all machines in your HugeCTR deployment. They allow you to take advantage of the combined memory capacity of your cluster machines.The configuration parameters for this kind of database should, thus, be identical across all achines in your deployment.

**Python**
```python
params = hugectr.inference.VolatileDatabaseParams(
type = "redis_cluster",
algorithm = hugectr.DatabaseHashMapAlgorithm_t.<enum_value>,
num_partitions = <integer_value>,
address = "<host_name_or_ip_address:port_number>",
user_name = "<login_user_name>",
password = "<login_password>",
num_partitions = <int_value>,
max_get_batch_size = <int_value>,
max_set_batch_size = <int_value>,
overflow_margin = <integer_value>,
overflow_policy = hugectr.DatabaseOverflowPolicy_t.<enum_value>,
overflow_resolution_target = <double_value>,
initial_cache_rate = <double_value>,
refresh_time_after_fetch = <True|False>,
cache_missed_embeddings = <True|False>,
update_filters = [ "<filter 0>", "<filter 1>", ... ]
)
```

**JSON**
```text
"volatile_db": {
  "type": "redis_cluster",
  "algorithm": "<enum_value>",
  "num_partitions": <integer_value>,
  "address": "<host_name_or_ip_address:port_number>",
  "user_name":  "<login_user_name>",
  "password": "<login_password>",
  "num_partitions": <integer_value>,
  "max_get_batch_size": <integer_value>,
  "max_set_batch_size": <integer_value>,
  "overflow_margin": <integer_value>,
  "overflow_policy": "<overflow_policy>",
  "overflow_resolution_target": <overflow_resolution_target>, 
  "initial_cache_rate": <double_value>, 
  "refresh_time_after_fetch": <true|false>, 
  "cache_missed_embeddings": <true|false>, 
  "update_filters": [ "<filter 0>", "<filter 1>", /* ... */ ]
  // ...
}
```
**Arguments:**
* `type`: use to specify which volatile database implementation to use. The `<enum_value>` is either:
    * `disabled`: Do not use this kind of database.
    * `hash_map`: Hash-map based CPU memory database implementation.
    * `parallel_hash_map`: Hash-map based CPU memory database implementation with multi threading support **(default)**.
    * `redis_cluster`: Connect to an existing Redis cluster deployment (Distributed CPU memory database implementation).

**The followings are Hashmap/Parallel Hashmap related parameters:**
* `algorithm`: use to specify the hashmap algorithm. Only for `hash_map` and `parallel_hash_map`. The `<enum_value>` is either:
    * `stl`: Use C++ standard template library-based hash-maps. This is a fallback implementation, that is generally less memory efficient and slower than `phm`. Use this, if you experience stability issues or problems with `phm`.
    * `phm`: Use use an [performance optimized hash-map implementation](https://greg7mdp.github.io/parallel-hashmap) **(default)**.

* `num_partitions`: integer. Only for `hash_map` and `parallel_hash_map`. Use to control the degree of parallelism. Parallel hash-map implementations split your embedding tables into roughly evenly sized partitions and parallelizes look-up and insert operations accordingly.
The **default value** is the equivalent to `min(number_of_cpu_cores, 16)` of the system that you used to build the HugeCTR binaries.

**The followings are Redis related parameters:**
* `address`: string, only for `redis_cluster`, the address of one of servers of the Redis cluster. format `"host_name[:port_number]"`, **default**: `"127.0.0.1:7000"`

* `user_name`: string, only for `redis_cluster`, the user name of our Redis cluster. **default**: `"default"` 

* `password`: string, only for `redis_cluster`, the password of your acount. **default**: `""`, *i.e.*, blank / no password.

* `num_partitions`: integer, only for `redis_cluster`. Our Redis cluster implementation breaks each embedding table into `num_partitions` approximately equal sized partitions. Each partition is assigned a storage location in your Redis cluster. We do not provide guarantees regarding the placement of partitions. Hence, multiple partitions might end up in the same node for some models and setups. Gernally speaking, to take advantage of your cluster resources, `num_partitions` need to be at least equal to the number of Redis nodes. For optimal performance `num_parititions` should be strictly larger than the amount of machines. However, each partition incurs a small processing overhead. So, the value should also not be too large. To retain a high performance and good cluster utilization, we **suggest** to adjust this value to 2-5x the number ofmachines in your Redis cluster. The **default value** is `8`.

* `max_get_batch_size` and `max_set_batch_size`: integer, only for `redis_cluster`, represent optimization parameters. Mass lookup and insert requests to distributed endpoints are chunked into batches. For maximum performance, these two parameters should be large. However, if the available memory for buffering requests in your endpoints is limited, or if you experience transmission stability issues, lowering this value may help. By **default**, both values are set to `10000`. With high-performance networking and endpoint hardware, it is **recommended** to increase these values to `1 million`.

**The followings are overflow handling related parameters:** 

To maximize performance and avoid instabilies caused by sporadic high memory usage (*i.e.*, out of memory situations), we provide the overflow handling mechanism. It allows limiting the maximum amount of embeddings to be stored per partition, and, thus, upper-bounding the memory consumption of your distributed database.

* `overflow_margin`: integer, denotes the maximum amount of embeddings that will be stored *per partition*. Inserting more than `overflow_margin` embeddings into the database will trigger the execution of the configured `overflow_policy`. Hence, `overflow_margin` upper-bounds the maximum amount of memory that your CPU memory database may occupy. Thumb rule: Larger `overflow_margin` will result higher hit rates, but also increased memory consumption. By **default**, the value of `overflow_margin` is set to `2^64 - 1` (*i.e.*, de-facto infinite). When using the CPU memory database in conjunction with a Persistent database, the idea value for `overflow_margin` may vary. In practice, a setting value to somewhere between `[1 million, 100 million]` tends deliver reliable performance and throughput.

* `overflow_policy`: can be either:
    * `evict_oldest` **(default)**: Prune embeddings starting from the oldest (i.e., least recently used) until the paratition contains at most `overflow_margin * overflow_resolution_target` embeddings.
    * `evict_random`: Prune embeddings random embeddings until the paratition contains at most `overflow_margin * overflow_resolution_target` embeddings.
    
    Unlike `evict_oldest`,  `evict_random` requires no comparison of time-stamps, and thus can be faster. However, `evict_oldest` is likely to deliver better performance over time because embeddings are evicted based on the frequency of their usage. 
    
* `overflow_resolution_target`: double, is expected to be in `]0, 1[` (*i.e.*, between `0` and `1`, but not exactly `0` or `1`). The default value of `overflow_resolution_target` is `0.8` (*i.e.*, the partition is shrunk to 80% of its maximum size, or in other words, when the partition size surpasses `overflow_margin` embeddings, 20% of the embeddings are evicted according to the respective `overflow_policy`).


* `initial_cache_rate`: double, should be the fraction (`[0.0, 1.0]`) of your dataset that we will attempt to cache immediately upon startup of the HPS database backend. Hence, setting a value of `0.5` causes the HugeCTR HPS database backend to attempt caching up to 50% of your dataset directly using the respectively configured volatile database after initialization.


**The following is a refreshing timestamps related parameter:**
* `refresh_time_after_fetch`: bool. Some algorithms to organize certain processes, such as the evication of embeddings upon overflow, take time into account. To evalute the affected embeddings, HugeCTR records the time when an embeddings is overridden. This is sufficient in training mode where embeddings are frequently replaced. Hence, the **default value** for this setting is is `false`. However, if you deploy HugeCTR only for inference (*e.g.*, with Triton), this might lead to suboptimal eviction patterns. By setting this value to `true`, HugeCTR will replace the time stored alongside an embedding right after this embedding is accessed. This operation may happen asynchronously (*i.e.*, with some delay).

**The following is related to caching of Missed Keys:**

* `cache_missed_embeddings`: bool, a value denoting whether or not to migrate embeddings into the volatile database if they were missed during lookup. Hence, if this value is set to `true`, an embedding that could not be retrieved from the volatile database, but could be retrieved from the persistent database, will be inserted into the volatile database - potentially replacing another value. The **default value** is `false`, which disables this functionality.

This feature will optimize the volatile database in response to the queries experienced in inference mode. In training mode, updated embeddings will be automatically written back to the databse after each training step. Thus, if you apply training, setting this setting to `true` will likely increase the number of writes to the database and degrade performance, without providing significant improvements, which is undesirable.

**The following is a real-time updating related parameter:**


* `update_filters`: this setting allows you specify a series of filters, in to permit / deny passing certain model updates from Kafka to the CPU memory database backend. Filters take the form of regular expressions. The **default** value of this setting is `[ "^hps_.+$" ]` (*i.e.*, process updates for all HPS models, irrespective of their name). **[Behavior will likely change in future versions]**

Distributed databases are shared by all your HugeCTR nodes. These nodes will collaborate to inject updates into the underlying database. The assignment of what nodes update what partition may change at runtime.

### **Persistent Database Configurations**
Persistent databases are instanced per machine and use the locally available non-volatile memory as backing storage. Hence, you may indvidually vary their configuration parameters per machine.

**Python**
```python
params = hugectr.inference.PersistentDatabaseParams(
  type = hugectr.DatabaseType_t.<enum_value>,
  path = "<file_system_path>",
  num_threads = <int_value>,
  read_only = <boolean_value>,
  max_get_batch_size = <int_value>,
  max_set_batch_size = <int_value>,
  update_filters = [ "<filter 0>", "<filter 1>", ... ]
)
```

**JSON**
```text
"persistent_db": {
  "type": "<enum_value>",
  "path": "<file_system_path>",
  "num_threads": <int_value>,
  "read_only": <boolean_value>,
  "max_get_batch_size": <int_value>,
  "max_set_batch_size": <int_value>,
  "update_filters": [ "<filter 0>", "<filter 1>", /* ... */ ]
}
```

* `type`:  is either:
    * `disabled`: Do not use this kind of database  **(default)**.
    * `rocks_db`: Create or connect to a RocksDB database.

* `path` denotes the directory in your file-system where the RocksDB database can be found. If the directory does not contain a RocksDB databse, HugeCTR will create an database for you. Note that this may override files that are currently stored in this database. Hence, make sure that `path` points either to an actual RocksDB database or an empty directy. The **default** path is `/tmp/rocksdb`.

* `num_threads` is an optimization parameter. This denotes the amount of threads that the RocksDB driver may use internally. By **default**, this value is set to `16`

* `read_only`, bool. If the flag `read_only` is set to `true`, the databse will be opened in *Read-Only mode*. Naturally, this means that any attempt to update values in this database will fail. Use for inference, if model is static and the database is shared by multiple nodes (for example via NFS). By **default** this flag is set to `false`.

* `max_get_batch_size` and `max_set_batch_size`, integer, represent optimization parameters. Mass lookup and insert requests to RocksDB are chunked into batches. For maximum performance `max_*_batch_size` should be large. However, if the available memory for buffering requests in your endpoints is limited, lowering this value may help. By **default**, both values are set to `10000`. With high-performance hardware setups it is **recommended** to increase these values to `1 million`.

* `update_filters`: this setting allows you specify a series of filters, in to permit / deny passing certain model updates from Kafka to the CPU memory database backend. Filters take the form of regular expressions. The **default** value of this setting is `[ "^hps_.+$" ]` (*i.e.*, process updates for all HPS models, irrespective of their name). **[Behavior will likely change in future versions]**

### **Update Source Configurations**
The real-time update source is the origin for model updates during online retraining. To ensure that all database layers are kept in sync, it is advisable configure all nodes in your HugeCTR deployment identical.

**Python**
```python
params = hugectr.UpdateSourceParams(
  type = hugectr.UpdateSourceType_t.<enum_value>,
  brokers = "host_name[:port][;host_name[:port]...]",
  metadata_refresh_interval_ms = <int_value>,
  receive_buffer_size = <int_value>,
  poll_timeout_ms = <int_value>,
  max_batch_size = <int_value>,
  failure_backoff_ms = <int_value>,
  max_commit_interval = <int_value>
)
```

**JSON**
```text
"update_source": {
  "type": "<enum_value>"
  "brokers": "host_name[:port][;host_name[:port]...]",
  "metadata_refresh_interval_ms": <int_value>,
  "receive_buffer_size": <int_value>,
  "poll_timeout_ms": <int_value>,
  "max_batch_size": <int_value>,
  "failure_backoff_ms": <int_value>,
  "max_commit_interval": <int_value>
}
```
**Arguments:**
* `type`: the update source implementation, is either:
    * `null`: Do not use this kind of database  **(default)**.
    * `kafka_message_queue`: Connect to an axisting Apache Kafka message queue.


* `brokers` *(string)*: In order to connect to a Kafka deployments, you need to fill in at least one host-address (hostname + port number) of a Kafka broker node (`brokers` configuration option in the above listings). The **default** value of `brokers` is `127.0.0.1:9092`.

* `metadata_refresh_interval_ms` *(integer)*: Frequency at which the topic metadata should be re-downloaded from the Kafka broker.

* `receive_buffer_size` *(integer)*: We allocate a buffer to temporarily store the data to be received from a Kafka. `receive_buffer_size` denotes the size of this buffer. This value should best match the `send_buffer_size` of the KafkaMessageSink that was used to push updates to Kafka. The **default** receive buffer size is `262144` bytes. *Note that the `message.max.bytes` setting of the Kafka broker must be at least `receive_buffer_size + 1024` bytes.*

* `poll_timeout_ms` *(integer)* denotes the maximum time we will wait for additional updates before dispatching them to the database layers in milliseconds. The **default** value is `500` ms.

* `max_batch_size` *(integer)*: Dispatching of updates is conducted in chunks. The maximum size of these chunks is upper-bounded by `max_batch_size`, which is set to `8192` by **default**.

* `failure_backoff_ms` *(integer)*: In some situations, there might be issues that prevent the successful dispatch of an update to a database. For example, if a Redis node is temporarily unreachable. `failure_backoff_ms` is the delay in milliseconds after which we retry dispatching a set of updates in such an event. The **default** backoff delay is `50` ms.
  
* `max_commit_interval` *(integer)*: Regardless of any other conditions, any received data will be forwarded and committed if at most `max_commit_interval` were processed since the previous commit. The **default** interval is `32`.

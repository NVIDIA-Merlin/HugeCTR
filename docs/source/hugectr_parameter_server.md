# HugeCTR Hierarchical Parameter Server

## Introduction

The hierarchical parameter server allows HugeCTR to use models with huge embedding tables by extending HugeCTRs storage space beyond the constraints of GPU memory through utilizing various memory resources across you cluster. Further, it grants the ability to permanently store embedding tables in a structured manner. For an end-to-end demo on how to use the hierarchical parameter server, please refer to [samples](https://github.com/triton-inference-server/hugectr_backend/tree/main/samples/hierarchical_deployment).


## Background

GPU clusters offer superior compute power, compared to their CPU-only counterparts. However, although modern data-center GPUs by NVIDIA are equipped with increasing amounts of memory, new and more powerful AI algorithms come into existence that require more memory. Recommendation models with their huge embedding tables are spearheading these developments. The parameter server allows you to efficiently train and perform inference with models that rely on embedding tables that vastly exceed the available GPU device storage space.

This is achieved through utilizing other memory resources, available within your clsuter, such as CPU accessible RAM and non-volatile memory. Aside from general advantages of non-volatile memory with respect to retaining stored information, storage devices such as HDDs and SDDs offer orders of magnitude more storage space than DDR memory and HBM (High Bandwidth Memory), at significantly lower cost. However, their throughout is lower and latency is higher than that of DRR and HBM.

The parameter server acts as an intermediate layer between your GPU and non-volatile memory to store all embeddings of your model. Thereby, using local RAM and/or RAM resources available accross the cluster as a cache to improve response times.


## Architecture

As of version 3.3, the HugeCTR parameter server defines 3 storage layers.

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

Models deployed viat the HugeCTR parameter server allow streaming model parameter updates from external sources via [Apache Kafka](https://kafka.apache.org). This function allows zero-downtime online model re-training - for example using the HugeCTR model training system.


## Execution

### Inference

With respect to embedding lookups via the HugeCTR embedding cache and parameter server, the following logic applies. Whenever the HugeCTR inference engine receives a batch of model input parameters for inference, it will first determine the associated unique embedding keys and try to resolve these embeddings using the embedding cache. If there are cache misses, it will then turn to the parameter server to determine the embedding representations. The query sequence inside the parameter server queries its configured backends in the following order to fill in the missing embeddings:

1. Local / Remote CPU memory locations
2. Permanent storage

Hence, if configured, HugeCTR will first try to lookup missing embeddings in either a *CPU Memory Database* or *Distributed Database*. If and only if, there are still missing embedding representations after that, HugeCTR will turn to non-volatile memory (via the *Persistent Database*, which contains a copy of all existing embeddings) to find the corresponding embedding representations.

### Training

After a training iteration, model updates for updated embeddings are published via Kafka by the HugeCTR training process. The parameter server can be configured to automatically listen to change requests for certain models and will ingest these updates in its various database stages.


### Lookup optimization

If volatile memory resources, i.e. the *CPU Memory Database* and/or *Distributed Database*, are not sufficient to retain the entire model, the HugeCTR will attempt to minimize the avarage latency for lookup through managing these resources like cache using an LRU paradigm.


## Configuration

The HugeCTR parameter server and iterative update can be configured using 3 separate configuration objects. Namely, the `VolatileDatabaseParams` and `PersistentDatabaseParams` are used to configure the database backends of each parameter server instance. If you desire iterative or online model updating, you must also provide the `UpdateSourceParams` configuration object to link the parameter server instance with your Kafka reployment. These objects are part of the Python package `hugectr.inference`.

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

### Volatile Database

We provide various volatile database implementations. Generally speaking, two categories can be distinguished.

* **CPU memory databases** are instanced per machine and only use the locally available RAM memory as backing storage. Hence, you may indvidually vary their configuration parameters per machine.

* **Distributed CPU memory databases** are typically shared by all machines in your HugeCTR deployment. They allow you to take advantage of the combined memory capacity of your cluster machines.The configuration parameters for this kind of database should, thus, be identical across all achines in your deployment.


**Python:**
```python
params = hugectr.inference.VolatileDatabaseParams()
```

#### Implementation Selection

**Python:**
```python
params.type = hugectr.DatabaseType_t.<enum_value>
```
**JSON:**
```text
"volatile_db": {
  "type": "<enum_value>"
  // ...
}
```

Where `<enum_value>` is either:
* `disabled`: Do not use this kind of database.
* `hash_map`: Hash-map based CPU memory database implementation.
* `parallel_hash_map`: Hash-map based CPU memory database implementation with multi threading support **(default)**.
* `redis_cluster`: Connect to an existing Redis cluster deployment (Distributed CPU memory database implementation).


#### Configuration of Normal Hash-map Backend

**Python:**
```python
params.type = hugectr.DatabaseType_t.hash_map
params.algorithm = hugectr.DatabaseHashMapAlgorithm_t.<enum_value>
```
**JSON:**
```text
"volatile_db": {
  "type": "hash_map",
  "algorithm": "<enum_value>"
  // ...
}
```

Where `<enum_value>` is either:

* `stl`: Use C++ standard template library-based hash-maps. This is a fallback implementation, that is generally less memory efficient and slower than `phm`. Use this, if you experience stability issues or problems with `phm`.

* `phm`: Use use an [performance optimized hash-map implementation](https://greg7mdp.github.io/parallel-hashmap) **(default)**.

All other settings will be ignored.


#### Configuration of Parallelized Hash-map Backend

**Python:**
```python
params.type = hugectr.DatabaseType_t.parallel_hash_map
params.algorithm = hugectr.DatabaseHashMapAlgorithm_t.<enum_value>
params.num_partitions = <integer_value>
```
**JSON:**
```text
"volatile_db": {
  "type": "parallel_hash_map",
  "algorithm": "<enum_value>",
  "num_partitions": <integer_value>
  // ...
}
```

Where `<enum_value>` is either:

* `stl`: Use C++ standard template library-based hash-maps. This is a fallback implementation, that is generally less memory efficient and slower than `phm`. Use this, if you experience stability issues or problems with `phm`.

* `phm`: Use use an [performance optimized hash-map implementation](https://greg7mdp.github.io/parallel-hashmap) **(default)**.

Parallel hash-map implementations split your embedding tables into roughly evenly sized partitions and parallelizes look-up and insert operations accordingly. With `<integer_value>` you can control the degree of parallelism. The **default value** is the equivalent to `min(number_of_cpu_cores, 16)` of the system that you used to build the HugeCTR binaries.


#### Configuration of Redis Cluster Backend

**Python:**
```python
params.type = "redis_cluster"
params.address = "<host_name_or_ip_address:port_number>"
params.user_name = "<login_user_name>"
params.password = "<login_password>"
params.num_partitions = <int_value>
params.max_get_batch_size = <int_value>
params.max_set_batch_size = <int_value>
```
**JSON:**
```text
"volatile_db": {
  "type": "redis_cluster",
  "address": "<host_name_or_ip_address:port_number>",
  "user_name":  "<login_user_name>",
  "password": "<login_password>",
  "num_partitions": <integer_value>,
  "max_get_batch_size": <integer_value>,
  "max_set_batch_size": <integer_value>
  // ...
}
```

Interpreted if `type` is set to `redis_cluster`. In order to successfully let the HugeCTR parameter server to your Redis cluster you need to provide at least the `address` (format `"host_name[:port_number]"`; **default**: `"127.0.0.1:7000"`) of one of your Redis servers, a valid `user_name` (**default**: `"default"`) and their `password` (**default**: `""`, *i.e.*, blank / no password.

Our Redis cluster implementation breaks each embedding table into `num_partitions` approximately equal sized partitions. Each partition is assigned a storage location in your Redis cluster. We do not provide guarantees regarding the placement of partitions. Hence, multiple partitions might end up in the same node for some models and setups. Gernally speaking, to take advantage of your cluster resources, `num_partitions` need to be at least equal to the number of Redis nodes. For optimal performance `num_parititions` should be strictly larger than the amount of machines. However, each partition incurs a small processing overhead. So, the value should also not be too large. To retain a high performance and good cluster utilization, we **suggest** to adjust this value to 2-5x the number ofmachines in your Redis cluster. The **default value** is `8`.

`max_get_batch_size` and `max_set_batch_size` represent optimization parameters. Mass lookup and insert requests to distributed endpoints are chunked into batches. For maximum performance `max_*_batch_size` should be large. However, if the available memory for buffering requests in your endpoints is limited, or if you experience transmission stability issues, lowering this value may help. By **default**, both values are set to `10000`. With high-performance networking and endpoint hardware, it is **recommended** to increase these values to `1 million`.


#### Overflow Handling Related Parameters

To maximize performance and avoid instabilies caused by sporadic high memory usage (*i.e.*, out of memory situations), we provide the overflow handling mechanism. It allows limiting the maximum amount of embeddings to be stored per partition, and, thus, upper-bounding the memory consumption of your distributed database.

**Python:**
```python
params.overflow_margin = <integer_value>
params.overflow_policy = hugectr.DatabaseOverflowPolicy_t.<enum_value>
params.overflow_resolution_target = <double_value>
```
**JSON:**
```text
"volatile_db": {
  "overflow_margin": <integer_value>,
  "overflow_policy": "<overflow_policy>",
  "overflow_resolution_target": <overflow_resolution_target>
  // ...
}
```

`overflow_margin` denotes the maximum amount of embeddings that will be stored *per partition*. Inserting more than `overflow_margin` embeddings into the database will trigger the execution of the configured `overflow_policy`. Hence, `overflow_margin` upper-bounds the maximum amount of memory that your CPU memory database may occupy. Thumb rule: Larger `overflow_margin` will result higher hit rates, but also increased memory consumption. By **default**, the value of `overflow_margin` is set to `2^64 - 1` (*i.e.*, de-facto infinite). When using the CPU memory database in conjunction with a Persistent database, the idea value for `overflow_margin` may vary. In practice, a setting value to somewhere between `[1 million, 100 million]` tends deliver reliable performance and throughput.

Currently the following values for `overflow_policy` are supported:
* `evict_oldest` **(default)**: Prune embeddings starting from the oldest (i.e., least recently used) until the paratition contains at most `overflow_margin * overflow_resolution_target` embeddings.
* `evict_random`: Prune embeddings random embeddings until the paratition contains at most `overflow_margin * overflow_resolution_target` embeddings.

Unlike `evict_oldest`,  `evict_random` requires no comparison of time-stamps, and thus can be faster. However, `evict_oldest` is likely to deliver better performance over time because embeddings are evicted based on the frequency of their usage. For all eviction policies, `overflow_resolution_target` is expected to be in `]0, 1[` (*i.e.*, between `0` and `1`, but not exactly `0` or `1`). The default value of `overflow_resolution_target` is `0.8` (*i.e.*, the partition is shrunk to 80% of its maximum size, or in other words, when the partition size surpasses `overflow_margin` embeddings, 20% of the embeddings are evicted according to the respective `overflow_policy`).


#### Initial Caching

**Python:**
```python
params.initial_cache_rate = <double_value>
```
**JSON:**
```text
"volatile_db": {
  "initial_cache_rate": <double_value>
  // ...
}
```

This is the fraction (`[0.0, 1.0]`) of your dataset that we will attempt to cache immediately upon startup of the parameter server. Hence, setting a value of `0.5` causes the HugeCTR parameter server to attempt caching up to 50% of your dataset directly using the respectively configured volatile database after initialization.


#### Refreshing Timestamps

**Python:**
```python
params.refresh_time_after_fetch = <True|False>
```
```
**JSON:**
```text
"volatile_db": {
  "refresh_time_after_fetch": <true|false>
  // ...
}
```

Some algorithms to organize certain processes, such as the evication of embeddings upon overflow, take time into account. To evalute the affected embeddings, HugeCTR records the time when an embeddings is overridden. This is sufficient in training mode where embeddings are frequently replaced. Hence, the **default value** for this setting is is `false`. However, if you deploy HugeCTR only for inference (*e.g.*, with Triton), this might lead to suboptimal eviction patterns. By setting this value to `true`, HugeCTR will replace the time stored alongside an embedding right after this embedding is accessed. This operation may happen asynchronously (*i.e.*, with some delay).


#### Caching of Missed Keys

**Python:**
```python
params.cache_missed_embeddings = <True|False>
```
**JSON:**
```text
"volatile_db": {
  "cache_missed_embeddings": <true|false>
  // ...
}
```

A boolean value denoting whether or not to migrate embeddings into the volatile database if they were missed during lookup. Hence, if this value is set to `true`, an embedding that could not be retrieved from the volatile database, but could be retrieved from the persistent database, will be inserted into the volatile database - potentially replacing another value. The **default value** is `false`, which disables this functionality.

This feature will optimize the volatile database in response to the queries experienced in inference mode. In training mode, updated embeddings will be automatically written back to the databse after each training step. Thus, if you apply training, setting this setting to `true` will likely increase the number of writes to the database and degrade performance, without providing significant improvements, which is undesirable.


#### Real-time Updating

**Python:**
```python
params.update_filters = [ "<filter 0>", "<filter 1>", ... ]
```
**JSON:**
```text
"volatile_db": {
  "update_filters": [ "<filter 0>", "<filter 1>", /* ... */ ]
  // ...
}
```

**[Behavior will likely change in future versions]** This setting allows you specify a series of filters, in to permit / deny passing certain model updates from Kafka to the CPU memory database backend. Filters take the form of regular expressions. The **default** value of this setting is `[ ".+" ]` (*i.e.*, process updates for all models, irrespective of their name).


Distributed databases are shared by all your HugeCTR nodes. These nodes will collaborate to inject updates into the underlying database. The assignment of what nodes update what partition may change at runtime.


### Persistent Database

Persistent databases are instanced per machine and use the locally available non-volatile memory as backing storage. Hence, you may indvidually vary their configuration parameters per machine.

**Python:**
```python
params = hugectr.inference.PersistentDatabaseParams()
```

#### Database Type Selection

**Python:**
```python
params.type = hugectr.DatabaseType_t.<enum_value>
```
**JSON:**
```text
"persistent_db": {
  "type": "<enum_value>"
}
```

Where `<enum_value>` is either:
* `disabled`: Do not use this kind of database  **(default)**.
* `rocks_db`: Create or connect to a RocksDB database.


#### Configuration of RocksDB Database Backend

**Python:**
```python
params.type = hugectr.DatabaseType_t.rocks_db
params.path = "<file_system_path>"
params.num_threads = <int_value>
params.read_only = <boolean_value>
params.max_get_batch_size = <int_value>
params.max_set_batch_size = <int_value>
```
**JSON:**
```text
"persistent_db": {
  "type": "rocks_db",
  "path": "<file_system_path>",
  "num_threads": <int_value>,
  "read_only": <boolean_value>,
  "max_get_batch_size": <int_value>,
  "max_set_batch_size": <int_value>
}
```

`path` denotes the directory in your file-system where the RocksDB database can be found. If the directory does not contain a RocksDB databse, HugeCTR will create an database for you. Note that this may override files that are currently stored in this database. Hence, make sure that `path` points either to an actual RocksDB database or an empty directy. The **default** path is `/tmp/rocksdb`.

`num_threads` is an optimization parameter. This denotes the amount of threads that the RocksDB driver may use internally. By **default**, this value is set to `16`

If the flag `read_only` is set to `true`, the databse will be opened in *Read-Only mode*. Naturally, this means that any attempt to update values in this database will fail. Use for inference, if model is static and the database is shared by multiple nodes (for example via NFS). By **default** this flag is set to `false`.

`max_get_batch_size` and `max_set_batch_size` represent optimization parameters. Mass lookup and insert requests to RocksDB are chunked into batches. For maximum performance `max_*_batch_size` should be large. However, if the available memory for buffering requests in your endpoints is limited, lowering this value may help. By **default**, both values are set to `10000`. With high-performance hardware setups it is **recommended** to increase these values to `1 million`.


#### Real-time Updating

**Python:**
```python
params.update_filters = [ "<filter 0>", "<filter 1>", ... ]
```
**JSON:**
```text
"persistent_db": {
  "update_filters": [ "<filter 0>", "<filter 1>", /* ... */ ]
  // ...
}
```

**[Behavior will likely change in future versions]** This setting allows you specify a series of filters, in to permit / deny passing certain model updates from Kafka to the CPU memory database backend. Filters take the form of regular expressions. The **default value** of this setting is `[ ".+" ]` (*i.e.*, process updates for all models, irrespective of their name).

### Real-time Update Source

<a id="markdown-real-time-update-source" name="real-time-update-source"></a>

The real-time update source is the origin for model updates during online retraining. To ensure that all database layers are kept in sync, it is advisable configure all nodes in your HugeCTR deployment identical.

**Python:**
```python
params = hugectr.inference.UpdateSourceParams()
```

#### Update Source Type Selection

**Python:**
```python
params.type = hugectr.UpdateSourceType_t.<enum_value>
```
**JSON:**
```text
"update_source": {
   "type": "<enum_value>"
}
```

Where `<enum_value>` is either:
* `null`: Do not use this kind of database  **(default)**.
* `kafka_message_queue`: Connect to an axisting Apache Kafka message queue.


#### Configuration Parameters for Apache Kafka Update Sources

**Python:**
```python
params.type = hugectr.UpdateSourceType_t.kafka_message_queue
params.brokers = "host_name[:port][;host_name[:port]...]"
params.poll_timeout_ms = <int_value>
params.max_receive_buffer_size = <int_value>
params.max_batch_size <int_value>
params.failure_backoff_ms = <int_value>
```
**JSON:**
```text
"update_source": {
  "type": "kafka_message_queue",
  "brokers": "host_name[:port][;host_name[:port]...]",
  "poll_timeout_ms": <int_value>,
  "max_receive_buffer_size": <int_value>,
  "max_batch_size": <int_value>,
  "failure_backoff_ms": <int_value>
}
```

In order to connect to a Kafka deployments, you need to fill in at least one host-address (hostname + port number) of a Kafka broker node (`brokers` configuration option in the above listings). The **default** value of `brokers` is `127.0.0.1:9092`.

The remaining parameters control certain properties within the notification chain. In particular, `poll_timeout_ms` denotes the maximum time we will wait for additional updates before dispatching them to the database layers in milliseconds. The **default** value is `500` ms.

If, before this limit has run out, more than `max_receive_buffer_size` embedding updates have been received, we will also dispatch these updates immediately. The **default** receive buffer size is `2000`.

Dispatching of updates is conducted in chunks. The maximum size of these chunks is upper-bounded by `max_batch_size`, which is set to `1000` by default.

In some situations, there might be issues that prevent the successful dispatch of an update to a database. For example, if a Redis node is temporarily unreachable. `failure_backoff_ms` is the delay in milliseconds after which we retry dispatching a set of updates in such an event. The **default** backoff delay is `50` ms.

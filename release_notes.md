# Release Notes

## What's New in Version 4.3

  ```{important}
  In January 2023, the HugeCTR team plans to deprecate semantic versioning, such as `v4.3`.
  Afterward, the library will use calendar versioning only, such as `v23.01`.
  ```

+ **Support for BERT and Variants**:
This release includes support for BERT in HugeCTR.
The documentation includes updates to the [MultiHeadAttention](https://nvidia-merlin.github.io/HugeCTR/v4.3/api/hugectr_layer_book.html#multiheadattention-layer) layer and adds documentation for the [SequenceMask](https://nvidia-merlin.github.io/HugeCTR/v4.3/api/hugectr_layer_book.html#sequencemask-layer) layer.
For more information, refer to the [samples/bst](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v4.3/samples/bst) directory of the repository in GitHub.

+ **HPS Plugin for TensorFlow integration with TensorFlow-TensorRT (TF-TRT)**:
This release includes plugin support for integration with TensorFlow-TensorRT.
For sample code, refer to the [Deploy SavedModel using HPS with Triton TensorFlow Backend](https://nvidia-merlin.github.io/HugeCTR/v4.3/hps_tf/notebooks/hps_tensorflow_triton_deployment_demo.html) notebook.

+ **Deep & Cross Network Layer version 2 Support**:
This release includes support for Deep & Cross Network version 2.
For conceptual information, refer to <https://arxiv.org/abs/2008.13535>.
The documentation for the [MultiCross Layer](https://nvidia-merlin.github.io/HugeCTR/v4.3/api/hugectr_layer_book.html#multicross-layer) is updated.

+ **Enhancements to Hierarchical Parameter Server**:
  + RedisClusterBackend now supports TLS/SSL communication.
    For sample code, refer to the [Hierarchical Parameter Server Demo](https://nvidia-merlin.github.io/HugeCTR/v4.3/notebooks/hps_demo.html) notebook.
    The notebook is updated with step-by-step instructions to show you how to setup HPS to use Redis with (and without) encryption.
    The [Volatile Database Parameters](https://nvidia-merlin.github.io/HugeCTR/v4.3/hugectr_parameter_server.html#volatile-database-parameters) documentation for HPS is updated with the `enable_tls`, `tls_ca_certificate`, `tls_client_certificate`, `tls_client_key`, and `tls_server_name_identification` parameters.
  + MultiProcessHashMapBackend includes a bug fix that prevented configuring the shared memory size when using JSON file-based configuration.
  + On-device input keys are supported now so that an extra host-to-device copy is removed to improve performance.
  + A dependency on the XX-Hash library is removed.
    The library is no longer used by HugeCTR.
  + Added the static table support to the embedding cache.
    The static table is suitable when the embedding table can be placed entirely in GPU memory.
    In this case, the static table is more than three times faster than the embedding cache lookup.
    The static table does not support embedding updates.

+ **Support for New Optimizers**:
  + Added support for SGD, Momentum SGD, Nesterov Momentum, AdaGrad, RMS-Prop, Adam and FTRL optimizers for dynamic embedding table (DET).
    For sample code, refer to the `test_embedding_table_optimizer.cpp` file in the [test/utest/embedding_collection/](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v4.3/test/utest/embedding_collection) directory of the repository on GitHub.
  + Added support for the FTRL optimizer for dense networks.

+ **Data Reading from S3 for Offline Inference**:
In addition to reading during training, HugeCTR now supports reading data from remote file systems such as HDFS and S3 during offline inference by using the DataSourceParams API.
The [HugeCTR Training and Inference with Remote File System Example](https://nvidia-merlin.github.io/HugeCTR/v4.3/notebooks/training_and_inference_with_remote_filesystem.html) is updated to demonstrate the new functionality.

+ **Documentation Enhancements**:
  + The set up [instructions for running the example notebooks](https://nvidia-merlin.github.io/HugeCTR/v4.3/notebooks/index.html) are revised for clarity.
  + The example notebooks are also updated to show using a data preprocessing script that simplifies the user experience.
  + Documentation for the [MLP Layer](https://nvidia-merlin.github.io/HugeCTR/v4.3/api/hugectr_layer_book.html#mlp-layer) is new.
  + Several 2022 talks and blogs are added to the [HugeCTR Talks and Blogs](https://nvidia-merlin.github.io/HugeCTR/v4.3/hugectr_talks_blogs.html) page.

+ **Issues Fixed**:
  + The original CUDA device with NUMA bind before a call to some HugeCTR APIs is recovered correctly now.
    This issue sometimes lead to a problem when you mixed calls to HugeCTR and other CUDA enabled libraries.
  + Fixed the occasional CUDA kernel launch failure of embedding when installed HugeCTR with macro DEBUG.
  + Fixed an SOK build error that was related to TensorFlow v2.1.0 and higher.
    The issue was that the C++ API and C++ standard were updated to use C++17.
  + Fixed a CUDA 12 related compilation error.

+ **Known Issues**:
  + HugeCTR can lead to a runtime error if client code calls the RMM `rmm::mr::set_current_device_resource()` method or  `rmm::mr::set_current_device_resource()` method.
    The error is due to the Parquet data reader in HugeCTR also calling `rmm::mr::set_current_device_resource()`.
    As a result, the device becomes visible to other libraries in the same process.
    Refer to GitHub [issue #356](https://github.com/NVIDIA-Merlin/HugeCTR/issues/356) for more information.
    As a workaround, you can set environment variable `HCTR_RMM_SETTABLE` to `0` to prevent HugeCTR from setting a custom RMM device resource, if you know that `rmm::mr::set_current_device_resource()` is called by client code other than HugeCTR.
    But be cautious because the setting can reduce the performance of Parquet reading.
  + HugeCTR uses NCCL to share data between ranks and NCCL can require shared system memory for IPC and pinned (page-locked) system memory resources.
    If you use NCCL inside a container, increase these resources by specifying the following arguments when you start the container:

    ```shell
      -shm-size=1g -ulimit memlock=-1
    ```

    See also the NCCL [known issue](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data) and the GitHub [issue #243](https://github.com/NVIDIA-Merlin/HugeCTR/issues/243).
  + `KafkaProducers` startup succeeds even if the target Kafka broker is unresponsive.
    To avoid data loss in conjunction with streaming-model updates from Kafka, you have to make sure that a sufficient number of Kafka brokers are running, operating properly, and are reachable from the node where you run HugeCTR.
  + The number of data files in the file list should be greater than or equal to the number of data reader workers.
    Otherwise, different workers are mapped to the same file and data loading does not progress as expected.
  + Joint loss training with a regularizer is not supported.
  + Dumping Adam optimizer states to AWS S3 is not supported.



## What's New in Version 4.2

  ```{important}
  In January 2023, the HugeCTR team plans to deprecate semantic versioning, such as `v4.2`.
  Afterward, the library will use calendar versioning only, such as `v23.01`.
  ```

+ **Change to HPS with Redis or Kafka**:
This release includes a change to Hierarchical Parameter Server and affects deployments that use `RedisClusterBackend` or model parameter streaming with Kafka.
A third-party library that was used for HPS partition selection algorithm is replaced to improve performance.
The new algorithm can produce different partition assignments for volatile databases.
As a result, volatile database backends that retain data between application startup, such as the `RedisClusterBackend`, must be reinitialized.
Model streaming with Kafka is equally affected.
To avoid issues with updates, reset all respective queue offsets to the `end_offset` before you reinitialize the `RedisClusterBackend`.

+ **Enhancements to the Sparse Operation Kit in DeepRec**:
This release includes updates to the Sparse Operation Kit to improve the performance of the embedding variable lookup operation in DeepRec.
The API for the `lookup_sparse()` function is changed to remove the `hotness` argument.
The `lookup_sparse()` function is enhanced to calculate the number of non-zero elements dynamically.
For more information, refer to the [sparse_operation_kit directory](https://github.com/alibaba/DeepRec/tree/main/addons/sparse_operation_kit) of the DeepRec repository in GitHub.

+ **Enhancements to 3G Embedding**:
This release includes the following enhancements to 3G embedding:
  + The API is changed.
    The `EmbeddingPlanner` class is replaced with the `EmbeddingCollectionConfig` class.
    For examples of the API, see the tests in the [test/embedding_collection_test](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v4.2/test/embedding_collection_test) directory of the repository in GitHub.
  + The API is enhanced to support dumping and loading weights during the training process.
    The methods are `Model.embedding_dump(path: str, table_names: list[str])` and `Model.embedding_load(path: str, list[str])`.
    The `path` argument is a directory in file system that you can dump weights to or load weights from.
    The `table_names` argument is a list of embedding table names as strings.

+ **New Volatile Database Type for HPS**:
This release adds a `db_type` value of `multi_process_hash_map` to the Hierarchical Parameter Server.
This database type supports sharing embeddings across process boundaries by using shared memory and the `/dev/shm` device file.
Multiple processes running HPS can read and write to the same hash map.
For an example, refer to the [Hierarchcal Parameter Server Demo](./notebooks/hps_demo.ipynb) notebook.

+ **Enhancements to the HPS Redis Backend**:
In this release, the Hierarchical Parameter Server can open multiple connections in parallel to each Redis node.
This enhancement enables HPS to take advantage of overlapped processing optimizations in the I/O module of Redis servers.
In addition, HPS can now take advantage of Redis hash tags to co-locate embedding values and metadata.
This enhancement can reduce the number of accesses to Redis nodes and the number of per-node round trip communications that are needed to complete transactions.
As a result, the enhancement increases the insertion performance.

+ **MLPLayer is New**:
This release adds an MLP layer with the `hugectr.Layer_t.MLP` class.
This layer is very flexible and makes it easier to use a group of fused fully-connected layers and enable the related optimizations.
For each fused fully-connected layer in `MLPLayer`, the output dimension, bias, and activation function are all adjustable.
MLPLayer supports FP32, FP16 and TF32 data types.
For an example, refer to the [dgx_a100_mlp.py](https://github.com/NVIDIA-Merlin/HugeCTR/blob/v4.2/samples/dlrm/dgx_a100_mlp.py) in the `samples/dlrm` directory of the GitHub repository to learn how to use the layer.

+ **Sparse Operation Kit installable from PyPi**:
Version `1.1.4` of the Sparse Operation Kit is installable from PyPi in the [merlin-sok](https://pypi.org/project/merlin-sok/) package.

+ **Multi-task Model Support added to the ONNX Model Converter**:
This release adds support for multi-task models to the ONNX converter.
This release also includes an enhancement to the [preprocess_census.py](https://github.com/NVIDIA-Merlin/HugeCTR/blob/v4.2/samples/mmoe/preprocess_census.py) script in `samples/mmoe` directory of the GitHub repository.

+ **Issues Fixed**:
  + Using the HPS Plugin for TensorFlow with `MirroredStrategy` and running the [Hierarchical Parameter Server Demo](https://nvidia-merlin.github.io/HugeCTR/v4.2/hierarchical_parameter_server/notebooks/hierarchical_parameter_server_demo.html) notebook triggered an issue with [ReplicaContext](https://www.tensorflow.org/api_docs/python/tf/distribute/ReplicaContext) and caused a crash.
    The issue is fixed and resolves GitHub [issue #362](https://github.com/NVIDIA-Merlin/HugeCTR/issues/362).
  + The [4_nvt_process.py](https://github.com/NVIDIA-Merlin/HugeCTR/blob/v4.2/samples/din/utils/4_nvt_process.py) sample in the `samples/din/utils` directory of the GitHub repository is updated to use the latest NVTabular API.
    This update resolves GitHub [issue #364](https://github.com/NVIDIA-Merlin/HugeCTR/issues/364).
  + An illegal memory access related to 3G embedding and the [dgx_a100_ib_nvlink.py](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v4.2/samples/dlrm/dgx_a100_ib_nvlink.py) sample in the `samples/dlrm` directory of the GitHub repository is fixed.
  + An error in HPS with the `lookup_fromdlpack()` method is fixed.
    The error was related to calculating the number of keys and vectors from the corresponding DLPack tensors.
  + An error in the HugeCTR backend for Triton Inference Server is fixed.
    A crash was triggered when the initial size of the embedding cache is smaller than the allowed minimum size.
  + An error related to using a ReLU layer with an odd input size in mixed precision mode could trigger a crash.
    The issue is fixed.
  + An error related to using an asynchronous reader with the [AsyncParam](https://nvidia-merlin.github.io/HugeCTR/v4.2/api/python_interface.html?highlight=io_alignment#asyncparam-class) class and specifying an `io_alignment` value that is smaller than the block device sector size is fixed.
    Now, if the specified `io_alignment` value is smaller than the block device sector size, `io_alignment` is automatically set to the block device sector size.
  + Unreported memory leaks in the GRU layer and collectives are fixed.
  + Several broken documentation links related to HPS are fixed.

+ **Known Issues**:
  + HugeCTR uses NCCL to share data between ranks and NCCL can require shared system memory for IPC and pinned (page-locked) system memory resources.
    If you use NCCL inside a container, increase these resources by specifying the following arguments when you start the container:

    ```shell
      -shm-size=1g -ulimit memlock=-1
    ```

    See also the NCCL [known issue](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data) and the GitHub [issue](https://github.com/NVIDIA-Merlin/HugeCTR/issues/243).
  + `KafkaProducers` startup succeeds even if the target Kafka broker is unresponsive.
    To avoid data loss in conjunction with streaming-model updates from Kafka, you have to make sure that a sufficient number of Kafka brokers are running, operating properly, and are reachable from the node where you run HugeCTR.
  + The number of data files in the file list should be greater than or equal to the number of data reader workers.
    Otherwise, different workers are mapped to the same file and data loading does not progress as expected.
  + Joint loss training with a regularizer is not supported.
  + Dumping Adam optimizer states to AWS S3 is not supported.

## What's New in Version 4.1

+ **Simplified Interface for 3G Embedding Table Placement Strategy**:
3G embedding now provides an easier way for you to configure an embedding table placement strategy.
Instead of using JSON, you can configure the embedding table placement strategy by using function arguments.
You only need to provide the `shard_matrix`, `table_group_strategy`, and `table_placement_strategy` arguments.
With these arguments, 3G embedding can group different tables together and place them according to the `shard_matrix` argument.
For an example, refer to [dlrm_train.py](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v4.1/test/embedding_collection_test/dlrm_train.py) file in the `test/embedding_collection_test` directory of the repository on GitHub.
For comparison, refer to the [same file](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v4.0/test/embedding_collection_test/dlrm_train.py) from the v4.0 branch of the repository.

+ **New MMoE and Shared-Bottom Samples**:
This release includes a new shared-bottom model, an example program, preprocessing scripts, and updates to documentation.
For more information, refer to the `README.md`, `mmoe_parquet.py`, and other files in the [`samples/mmoe`](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v4.1/samples/mmoe) directory of the repository on GitHub.
This release also includes a fix to the calculation and reporting of AUC for multi-task models, such as MMoE.

+ **Support for AWS S3 File System**:
The Parquet DataReader can now read datasets from the Amazon Web Services S3 file system.
You can also load and dump models from and to S3 during training.
The documentation for the [`DataSourceParams`](https://nvidia-merlin.github.io/HugeCTR/v4.1/api/python_interface.html#datasourceparams-class) class is updated.
To view sample code, refer to the [HugeCTR Training with Remote File System Example](https://nvidia-merlin.github.io/HugeCTR/v4.1/notebooks/training_with_remote_filesystem.html) class is updated.

+ **Simplication for File System Usage**:
You no longer ’t need to pass `DataSourceParams` for model loading and dumping.
The `FileSystem` class automatically infers the correct file system type, local, HDFS, or S3, based on the path URI that you specified when you built the model.
For example, the path `hdfs://localhost:9000/` is inferred as an HDFS file system and the path `https://mybucket.s3.us-east-1.amazonaws.com/` is inferred as an S3 file system.

+ **Support for Loading Models from Remote File Systems to HPS**:
This release enables you to load models from HDFS and S3 remote file systems to HPS during inference.
To use the new feature, specify an HDFS for S3 path URI in `InferenceParams`.

+ **Support for Exporting Intermediate Tensor Values into a Numpy Array**:
This release adds function `check_out_tensor` to `Model` and `InferenceModel`.
You can use this function to check out the intermediate tensor values using the Python interface.
This function is especially helpful for debugging.
For more information, refer to [`Model.check_out_tensor`](https://nvidia-merlin.github.io/HugeCTR/v4.1/api/python_interface.html#check-out-tensor-method) and [`InferenceModel.check_out_tensor`](https://nvidia-merlin.github.io/HugeCTR/v4.1/api/python_interface.html#id3).

+ **On-Device Input Keys for HPS Lookup**:
The HPS lookup supports input embedding keys that are on GPU memory during inference.
This enhancement removes a host-to-device copy by using the DLPack `lookup_fromdlpack()` interface.
By using the interface, the input DLPack capsule of embedding key can be a GPU tensor.

+ **Documentation Enhancements**:
  + The graphic for the [Hierarchical Parameter Server](https://nvidia-merlin.github.io/HugeCTR/v4.1/hierarchical_parameter_server/index.html) library that shows relationship to other software packages is enhanced.
  + The sample notebook for [Deploy SavedModel using HPS with Triton TensorFlow Backend](https://nvidia-merlin.github.io/HugeCTR/v4.1/hierarchical_parameter_server/notebooks/hps_tensorflow_triton_deployment_demo.html) is added to the documentation.
  + Style updates to the [Hierarchical Parameter Server API](https://nvidia-merlin.github.io/HugeCTR/v4.1/hierarchical_parameter_server/api/index.html) documentation.

+ **Issues Fixed**:
  + The `InteractionLayer` class is fixed so that it works correctly with `num_feas > 30`.
  + The cuBLASLt configuration is corrected by increasing the workspace size and adding the epilogue mask.
  + The NVTabular based preprocessing script for our samples that demonstrate feature crossing is fixed.
  + The async data reader is fixed. Previously, it would hang and cause a corruption issue due to an improper I/O block size and I/O alignment problem.
    The `AsyncParam` class is changed to implement the fix.
    The `io_block_size` argument is replaced by the `max_nr_request` argument and the actual I/O block size that the async reader uses is computed accordingly.
    For more information, refer to the [`AsyncParam`](https://nvidia-merlin.github.io/HugeCTR/v4.1/api/python_interface.html#asyncparam) class documentation.

+ **Known Issues**:
  + HugeCTR uses NCCL to share data between ranks and NCCL can require shared system memory for IPC and pinned (page-locked) system memory resources.
    If you use NCCL inside a container, increase these resources by specifying the following arguments when you start the container:

    ```shell
      -shm-size=1g -ulimit memlock=-1
    ```

    See also the NCCL [known issue](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data) and the GitHub [issue](https://github.com/NVIDIA-Merlin/HugeCTR/issues/243).
  + `KafkaProducers` startup succeeds even if the target Kafka broker is unresponsive.
    To avoid data loss in conjunction with streaming-model updates from Kafka, you have to make sure that a sufficient number of Kafka brokers are running, operating properly, and are reachable from the node where you run HugeCTR.
  + The number of data files in the file list should be greater than or equal to the number of data reader workers.
    Otherwise, different workers are mapped to the same file and data loading does not progress as expected.
  + Joint loss training with a regularizer is not supported.
  + Dumping Adam optimizer states to AWS S3 is not supported.
  + Dumping to remote file systems when enabled MPI is not supported.

## What's New in Version 4.0

+ **3G Embedding Stablization**:
Since the introduction of the next generation of HugeCTR embedding in v3.7, several updates and enhancements were made, including code refactoring to improve usability.
The enhancements for this release are as follows:
  + Optimized the performance for sparse lookup in terms of inter-warp load imbalance.
    Sparse Operation Kit (SOK) takes advantage of the enhancement to improve performance.
  + This release includes a fix for determining the maximum embedding vector size in the `GlobalEmbeddingData` and `LocalEmbeddingData` classes.
  + Version 1.1.4 of Sparse Operation Kit can be installed with Pip and includes the enhancements mentioned in the preceding bullets.

+ **Embedding Cache Initialization with Configurable Ratio**:
In previous releases, the default value for the `cache_refresh_percentage_per_iteration` parameter of the [InferenceParams](https://nvidia-merlin.github.io/HugeCTR/v4.0/hugectr_parameter_server.html#inference-parameters-and-embedding-cache-configuration) was `0.1`.

  In this release, default value is `0.0` and the parameter provides an additional purpose.
  If you set the parameter to a value greater than `0.0` and also set `use_gpu_embedding_cache` to `True` for a model, when Hierarchical Parameter Server (HPS) starts, HPS initializes the embedding cache for the model on the GPU by loading a subset of the embedding vectors from the sparse files for the model.
  When embedding cache initialization is used, HPS creates log records when it starts at the INFO level.
  The logging records are similar to `EC initialization for model: "<model-name>", num_tables: <int>` and `EC initialization on device: <int>`.
  This enhancement reduces the duration of the warm up phase.

+ **Lazy Initialization of HPS Plugin for TensorFlow**:
In this release, when you deploy a `SavedModel` of TensorFlow with Triton Inference Server, HPS is implicitly initialized when the loaded model is executed for the first time.
In previous releases, you needed to run `hps.Init(ps_config_file, global_batch_size)` explicitly.
For more information, see the API documentation for [`hierarchical_parameter_server.Init`](https://nvidia-merlin.github.io/HugeCTR/v4.0/hierarchical_parameter_server/api/initialize.html#hierarchical_parameter_server.Init).

+ **Enhancements to the HDFS Backend**:
  + The HDFS Backend is now called IO::HadoopFileSystem.
  + This release includes fixes for memory leaks.
  + This release includes refactoring to generalize the interface for HDFS and S3 as remote filesystems.
  + For more information, see `hadoop_filesystem.hpp` in the [`include/io`](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v4.0/HugeCTR/include/io) directory of the repository on GitHub.

+ **Dependency Clarification for Protobuf and Hadoop**:
Hadoop and Protobuf are true `third_party` modules now.
Developers can now avoid unnecessary and frequent cloning and deletion.

+ **Finer granularity control for overlap behavior**:
We deperacated the old `overlapped_pipeline` knob and introduces four new knobs `train_intra_iteration_overlap`/`train_inter_iteration_overlap`/`eval_intra_iteration_overlap`/`eval_inter_iteration_overlap` to help user better control the overlap behavior. For more information, see the API documentation for [`Solver.CreateSolver`](https://nvidia-merlin.github.io/HugeCTR/v4.0/api/python_interface.html#createsolver-method)

+ **Documentation Improvements**:
  + Removed two deprecated tutorials `triton_tf_deploy` and `dump_to_tf`.
  + Previously, the graphics in the [Performance](https://nvidia-merlin.github.io/HugeCTR/v4.0/performance.html) page did not appear.
    This issue is fixed in this release.
  + Previously, the [API documentation](https://nvidia-merlin.github.io/HugeCTR/v4.0/hierarchical_parameter_server/api/index.html) for the HPS Plugin for TensorFlow did not show the class information. This issue is fixed in this release.


+ **Issues Fixed**:
  + Fixed a build error that was triggered in debug mode.
    The error was caused by the newly introduced 3G embedding unit tests.
  + When using the Parquet DataReader, if a parquet dataset file specified in `metadata.json` does not exist, HugeCTR no longer crashes.
    The new behavior is to skip the missing file and display a warning message.
    This change relates to GitHub issue [321](https://github.com/NVIDIA-Merlin/HugeCTR/issues/321).

+ **Known Issues**:
  + HugeCTR uses NCCL to share data between ranks and NCCL can require shared system memory for IPC and pinned (page-locked) system memory resources.
    If you use NCCL inside a container, increase these resources by specifying the following arguments when you start the container:

    ```shell
      -shm-size=1g -ulimit memlock=-1
    ```

    See also the NCCL [known issue](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data) and the GitHub [issue](https://github.com/NVIDIA-Merlin/HugeCTR/issues/243).
  + `KafkaProducers` startup succeeds even if the target Kafka broker is unresponsive.
    To avoid data loss in conjunction with streaming-model updates from Kafka, you have to make sure that a sufficient number of Kafka brokers are running, operating properly, and are reachable from the node where you run HugeCTR.
  + The number of data files in the file list should be greater than or equal to the number of data reader workers.
    Otherwise, different workers are mapped to the same file and data loading does not progress as expected.
  + Joint loss training with a regularizer is not supported.


## What's New in Version 3.9

+ **Updates to 3G Embedding**:
  + Sparse Operation Kit (SOK) is updated to use the HugeCTR 3G embedding as a developer preview feature.
    For more information, refer to the Python programs in the [sparse_operation_kit/experiment/benchmark/dlrm](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.9/sparse_operation_kit/sparse_operation_kit/experiment/benchmark/dlrm) directory of the repository on GitHub.
  + Dynamic embedding table mode is added.
    The mode is based on the [cuCollection](https://github.com/NVIDIA/cuCollections) with some functionality enhancement.
    A dynamic embedding table grows its size when the table is full so that you no longer need to configure the memory usage information for embedding.
    For more information, refer to the [embedding_storage/dynamic_embedding_storage](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.9/HugeCTR/embedding_storage/dynamic_embedding_table) directory of the repository on GitHub.

+ **Enhancements to the HPS Plugin for TensorFlow**:
This release includes improvements to the interoperability of SOK and HPS.
The plugin now supports the sparse lookup layer.
The documentation for the HPS plugin is enhanced as follows:
  + An [introduction](https://nvidia-merlin.github.io/HugeCTR/v3.9/hierarchical_parameter_server/hps_tf_user_guide.html) to the plugin is new.
  + New [notebooks](https://nvidia-merlin.github.io/HugeCTR/v3.9/hierarchical_parameter_server/notebooks/index.html) demonstrate how to use the HPS plugin are added.
  + [API documentation](https://nvidia-merlin.github.io/HugeCTR/v3.9/hierarchical_parameter_server/api/index.html) for the plugin is new.

+ **Enhancements to the HPS Backend for Triton Inference Server**
This release adds support for integrating the [HPS Backend](https://github.com/triton-inference-server/hugectr_backend/tree/main/hps_backend) and the [TensorFlow Backend](https://github.com/triton-inference-server/tensorflow_backend) through the ensemble mode with Triton Inference Server.
The enhancement enables deploying a TensorFlow model with large embedding tables with Triton by leveraging HPS.
For more information, refer to the sample programs in the [hps-triton-ensemble](https://github.com/triton-inference-server/hugectr_backend/tree/main/hps_backend/samples/hps-triton-ensemble) directory of the HugeCTR Backend repository in GitHub.

+ **New Multi-Node Tutorial**:
The [multi-node training tutorial](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.9/tutorial/multinode-training) is new.
The additions show how to use HugeCTR to train a model with multiple nodes and is based on our most recent Docker container.
The tutorial should be useful to users who do not have a job-scheduler-installed cluster such as Slurm Workload Manager.
The update addresses a issue that was first reported in GitHub issue [305](https://github.com/NVIDIA-Merlin/HugeCTR/issues/305).

+ **Support Offline Inference for MMoE**:
This release includes MMoE offline inference where both per-class AUC and average AUC are provided.
When the number of class AUCs is greater than one, the output includes a line like the following example:

   ```console
   [HCTR][08:52:59.254][INFO][RK0][main]: Evaluation, AUC: {0.482141, 0.440781}, macro-averaging AUC: 0.46146124601364136
   ```

+ **Enhancements to the API for the HPS Database Backend**
This release includes several enhancements to the API for the `DatabaseBackend` class.
For more information, see `database_backend.hpp` and the header files for other database backends in the [`HugeCTR/include/hps`](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.9/HugeCTR/include/hps) directory of the repository.
The enhancements are as follows:
  + You can now specify a maximum time budget, in nanoseconds, for queries so that you can build an application that must operate within strict latency limits.
    Fetch queries return execution control to the caller if the time budget is exhausted.
    The unprocessed entries are indicated to the caller through a callback function.
  + The `dump` and `load_dump` methods are new.
    These methods support saving and loading embedding tables from disk.
    The methods support a custom binary format and the RocksDB SST table file format.
    These methods enable you to import and export embedding table data between your custom tools and HugeCTR.
  + The `find_tables` method is new.
    The method enables you to discover all table data that is currently stored for a model in a `DatabaseBackend` instance.
    A new overloaded method for `evict` is added that can process the results from `find_tables` to quickly and simply drop all the stored information that is related to a model.

+ **Documentation Enhancements**
  + The documentation for the `max_all_to_all_bandwidth` parameter of the [`HybridEmbeddingParam`](https://nvidia-merlin.github.io/HugeCTR/v3.9/api/python_interface.html#hybridembeddingparam-class) class is clarified to indicate that the bandwidth unit is per-GPU.
    Previously, the unit was not specified.


+ **Issues Fixed**:
  + Hybrid embedding with `IB_NVLINK` as the `communication_type` of the
    [`HybridEmbeddingParam`](https://nvidia-merlin.github.io/HugeCTR/v3.9/api/python_interface.html#hybridembeddingparam-class)
    is fixed in this release.
  + Training performance is affected by a GPU routine that checks if an input key can be out of the embedding table. If you can guarantee that the input keys can work with the specified `workspace_size_per_gpu_in_mb`, we have a workaround to disable the routine by setting the environment variable `HUGECTR_DISABLE_OVERFLOW_CHECK=1`. The workaround restores the training performance.
  + Engineering discovered and fixed a correctness issue with the Softmax layer.
  + Engineering removed an inline profiler that was rarely used or updated. This change relates to GitHub issue [340](https://github.com/NVIDIA-Merlin/HugeCTR/issues/340).

+ **Known Issues**:
  + HugeCTR uses NCCL to share data between ranks and NCCL can require shared system memory for IPC and pinned (page-locked) system memory resources.
    If you use NCCL inside a container, increase these resources by specifying the following arguments when you start the container:

    ```shell
      -shm-size=1g -ulimit memlock=-1
    ```

    See also the NCCL [known issue](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data) and the GitHub [issue](https://github.com/NVIDIA-Merlin/HugeCTR/issues/243).
  + `KafkaProducers` startup succeeds even if the target Kafka broker is unresponsive.
    To avoid data loss in conjunction with streaming-model updates from Kafka, you have to make sure that a sufficient number of Kafka brokers are running, operating properly, and are reachable from the node where you run HugeCTR.
  + The number of data files in the file list should be greater than or equal to the number of data reader workers.
    Otherwise, different workers are mapped to the same file and data loading does not progress as expected.
  + Joint loss training with a regularizer is not supported.


## What's New in Version 3.8

+ **Sample Notebook to Demonstrate 3G Embedding**:
This release includes a sample notebook that introduces the Python API of the
embedding collection and the key concepts for using 3G embedding.
You can view [HugeCTR Embedding Collection](https://nvidia-merlin.github.io/HugeCTR/v3.8/notebooks/embedding_collection.html)
from the documentation or access the `embedding_collection.ipynb` file from the
[`notebooks`](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.8/notebooks/)
directory of the repository.

+ **DLPack Python API for Hierarchical Parameter Server Lookup**:
This release introduces support for embedding lookup from the Hierarchical
Parameter Server (HPS) using the DLPack Python API.  The new method is
`lookup_fromdlpack()`.  For sample usage, see the
[Lookup the Embedding Vector from DLPack](https://nvidia-merlin.github.io/HugeCTR/v3.8/notebooks/hps_demo.html#lookup-the-embedding-vector-from-dlpack)
heading in the "Hierarchical Parameter Server Demo" notebook.

+ **Read Parquet Datasets from HDFS with the Python API**:
This release enhances the [`DataReaderParams`](https://nvidia-merlin.github.io/HugeCTR/v3.8/api/python_interface.html#datareaderparams)
class with a `data_source_params` argument. You can use the argument to specify
the data source configuration such as the host name of the Hadoop NameNode and the NameNode port number to read from HDFS.

+ **Logging Performance Improvements**:
This release includes a performance enhancement that reduces the performance impact of logging.

+ **Enhancements to Layer Classes**:
  + The `FullyConnected` layer now supports 3D inputs
  + The `MatrixMultiply` layer now supports 4D inputs.

+ **Documentation Enhancements**:
  + An automatically generated table of contents is added to the top of most
    pages in the web documentation. The goal is to provide a better experience
    for navigating long pages such as the
    [HugeCTR Layer Classes and Methods](https://nvidia-merlin.github.io/HugeCTR/v3.8/api/hugectr_layer_book.html)
    page.
  + URLs to the Criteo 1TB click logs dataset are updated. For an example, see the
    [HugeCTR Wide and Deep Model with Criteo](https://nvidia-merlin.github.io/HugeCTR/v3.8/notebooks/hugectr_wdl_prediction.html)
    notebook.

+ **Issues Fixed**:
  + The data generator for the Parquet file type is fixed and produces consistent file names between the `_metadata.json` file and the actual dataset files.
    Previously, running the data generator to create synthetic data resulted in a core dump.
    This issue was first reported in the GitHub issue [321](https://github.com/NVIDIA-Merlin/HugeCTR/issues/321).
  + Fixed the memory crash in running a large model on multiple GPUs that occurred during AUC warm up.
  + Fixed the issue of keyset generation in the ETC notebook.
    Refer to the GitHub issue [332](https://github.com/NVIDIA-Merlin/HugeCTR/issues/332) for more details.
  + Fixed the inference build error that occurred when building with debug mode.
  + Fixed the issue that multi-node training prints duplicate messages.

+ **Known Issues**:
  + Hybrid embedding with `IB_NVLINK` as the `communication_type` of the
    [`HybridEmbeddingParam`](https://nvidia-merlin.github.io/HugeCTR/v3.8/api/python_interface.html#hybridembeddingparam-class)
    class does not work currently. We are working on fixing it. The other communication types have no issues.
  + HugeCTR uses NCCL to share data between ranks and NCCL can require shared system memory for IPC and pinned (page-locked) system memory resources.
    If you use NCCL inside a container, increase these resources by specifying the following arguments when you start the container:

    ```shell
      -shm-size=1g -ulimit memlock=-1
    ```

    See also the NCCL [known issue](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data) and the GitHub [issue](https://github.com/NVIDIA-Merlin/HugeCTR/issues/243).
  + `KafkaProducers` startup succeeds even if the target Kafka broker is unresponsive.
    To avoid data loss in conjunction with streaming-model updates from Kafka, you have to make sure that a sufficient number of Kafka brokers are running, operating properly, and are reachable from the node where you run HugeCTR.
  + The number of data files in the file list should be greater than or equal to the number of data reader workers.
    Otherwise, different workers are mapped to the same file and data loading does not progress as expected.
  + Joint loss training with a regularizer is not supported.

## What's New in Version 3.7

+ **3G Embedding Developer Preview**:
The 3.7 version introduces next-generation of embedding as a developer preview feature. We call it 3G embedding because it is the new update to the HugeCTR embedding interface and implementation since the unified embedding in v3.1 version, which was the second one.
Compared with the previous embedding, there are three main changes in the embedding collection.
  + First, it allows users to fuse embedding tables with different embedding vector sizes. The previous embedding can only fuse embedding tables with the same embedding vector size.
  The enhancement boosts both flexibility and performance.
  + Second, it extends the functionality of embedding by supporting the `concat` combiner and supporting different slot lookup on the same embedding table.
  + Finally, the embedding collection is powerful enough to support arbitrary embedding table placement which includes data parallel and model parallel.
  By providing a plan JSON file, you can configure the table placement strategy as you specify.
  See the `dlrm_train.py` file in the  [embedding_collection_test](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.7/test/embedding_collection_test) directory of the repository for a more detailed usage example.

+ **HPS Performance Improvements**:
  + **Kafka**: Model parameters are now stored in Kafka in a bandwidth-saving multiplexed data format.
  This data format vastly increases throughput. In our lab, we measured transfer speeds up to 1.1 Gbps for each Kafka broker.
  + **HashMap backend**: Parallel and single-threaded hashmap implementations have been replaced by a new unified implementation.
  This new implementation uses a new memory-pool based allocation method that vastly increases upsert performance without diminishing recall performance.
  Compared with the previous implementation, you can expect a 4x speed improvement for large-batch insertion operations.
  + **Suppressed and simplified log**: Most log messages related to HPS have the log level changed to `TRACE` rather than `INFO` or `DEBUG` to reduce logging verbosity.

+ **Offline Inference Usability Enhancements**:
  + The thread pool size is configurable in the Python interface, which is useful for studying the embedding cache performance in scenarios of asynchronous update. Previously it was set as the minimum value of 16 and `std::thread::hardware_concurrency()`. For more information, please refer to [Hierarchical Parameter Server Configuration](https://nvidia-merlin.github.io/HugeCTR/v3.7/hugectr_parameter_server.html#configuration).


+ **DataGenerator Performance Improvements**:
You can specify the `num_threads` parameter to parallelize a `Norm` dataset generation.

+ **Evaluation Metric Improvements**:
  + Average loss performance improvement in multi-node environments.
  + AUC performance optimization and safer memory management.
  + Addition of NDCG and SMAPE.

+ **Embedding Training Cache Parquet Demo**:
Created a keyset extractor script to generate keyset files for Parquet datasets.
Provided users with an end-to-end demo of how to train a Parquet dataset using the embedding cache mode.
See the [Embedding Training Cache Example](https://nvidia-merlin.github.io/HugeCTR/v3.7/notebooks/embedding_training_cache_example.html) notebook.

+ **Documentation Enhancements**:
The documentation details for [HugeCTR Hierarchical Parameter Server Database Backend](https://nvidia-merlin.github.io/HugeCTR/v3.7/hugectr_parameter_server.html) are updated for consistency and clarity.

+ **Issues Fixed**:
  + If `slot_size_array` is specified, `workspace_size_per_gpu_in_mb` is no longer required.
  + If you build and install HugeCTR from scratch, you can specify the `CMAKE_INSTALL_PREFIX` CMake variable to identify the installation directory for HugeCTR.
  + Fixed SOK hang issue when calling `sok.Init()` with a large number of GPUs. See the GitHub issue [261](https://github.com/NVIDIA-Merlin/HugeCTR/issues/261) and [302](https://github.com/NVIDIA-Merlin/HugeCTR/issues/302) for more details.


+ **Known Issues**:
  + HugeCTR uses NCCL to share data between ranks and NCCL can require shared system memory for IPC and pinned (page-locked) system memory resources.
    If you use NCCL inside a container, increase these resources by specifying the following arguments when you start the container:

    ```shell
      -shm-size=1g -ulimit memlock=-1
    ```

    See also the NCCL [known issue](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data) and the GitHub [issue](https://github.com/NVIDIA-Merlin/HugeCTR/issues/243).
  + `KafkaProducers` startup succeeds even if the target Kafka broker is unresponsive.
    To avoid data loss in conjunction with streaming-model updates from Kafka, you have to make sure that a sufficient number of Kafka brokers are running, operating properly, and are reachable from the node where you run HugeCTR.
  + The number of data files in the file list should be greater than or equal to the number of data reader workers.
    Otherwise, different workers are mapped to the same file and data loading does not progress as expected.
  + Joint loss training with a regularizer is not supported.
  + The Criteo 1 TB click logs dataset that is used with many HugeCTR sample programs and notebooks is currently unavailable.
    Until the dataset becomes downloadable again, you can run those samples based on our synthetic dataset generator.
    For more information, see the [Getting Started](https://github.com/NVIDIA-Merlin/HugeCTR#getting-started) section of the repository README file.
  + Data generator of parquet type produces inconsistent file names between _metadata.json and actual dataset files, which will result in core dump fault when using the synthetic dataset.

## What's New in Version 3.6

+ **Concat 3D Layer**:
In previous releases, the `Concat` layer could handle two-dimensional (2D) input tensors only.
Now, the input can be three-dimensional (3D) and you can concatenate the inputs along axis 1 or 2.
For more information, see the API documentation for the [Concat Layer](https://nvidia-merlin.github.io/HugeCTR/v3.6/api/hugectr_layer_book.html#concat-layer).

+ **Dense Column List Support in Parquet DataReader**:
In previous releases, HugeCTR assumes each dense feature has a single value and it must be the scalar data type `float32`.
Now, you can mix `float32` or `list[float32]` for dense columns.
This enhancement means that each dense feature can have more than one value.
For more information, see the API documentation for the [Parquet](https://nvidia-merlin.github.io/HugeCTR/v3.6/api/python_interface.html#parquet) dataset format.

+ **Support for HDFS is Re-enabled in Merlin Containers**:
Support for HDFS in Merlin containers is an optional dependency now.
For more information, see [HDFS Support](https://nvidia-merlin.github.io/HugeCTR/v3.6/hugectr_core_features.html#hdfs-support).

+ **Evaluation Metric Enhancements**:
In previous releases, HugeCTR computes AUC for binary classification only.
Now, HugeCTR supports AUC for multi-label classification.
The implementation is inspired by [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) and performs the unweighted macro-averaging strategy that is the default for scikit-learn.
You can specify a value for the `label_dim` parameter of the input layer to enable multi-label classification and HugeCTR will compute the multi-label AUC.

+ **Log Output Format Change**:
The default log format now includes milliseconds.

+ **Documentation Enhancements**:
  + These release notes are included in the documentation and are available at <https://nvidia-merlin.github.io/HugeCTR/v3.6/release_notes.html>.
  + The [Configuration](https://nvidia-merlin.github.io/HugeCTR/v3.6/hugectr_parameter_server.html#configuration) section of the Hierarchical Parameter Server information is updated with more information about the parameters in the configuration file.
  + The example notebooks that demonstrate how to work with multi-modal data are reorganized in the navigation.
    The notebooks are now available under the heading [Multi-Modal Example Notebooks](https://nvidia-merlin.github.io/HugeCTR/v3.6/notebooks/multi-modal-data/index.html).
    This change is intended to make it easier to find the notebooks.
  + The documentation in the [sparse_operation_kit](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.6/sparse_operation_kit) directory of the repository on GitHub is updated with several clarifications about SOK.

+ **Issues Fixed**:
  + The `dlrm_kaggle_fp32.py` file in the [`samples/dlrm/`](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.6/samples/dlrm) directory of the repository is updated to show the correct number of samples.
    The `num_samples` value is now set to `36672493`.
    This fixes GitHub issue [301](https://github.com/NVIDIA-Merlin/HugeCTR/issues/301).
  + Hierarchical Parameter Server (HPS) would produce a runtime error when the GPU cache was turned off.
    This issue is now fixed.

+ **Known Issues**:
  + HugeCTR uses NCCL to share data between ranks and NCCL can require shared system memory for IPC and pinned (page-locked) system memory resources.
    If you use NCCL inside a container, increase these resources by specifying the following arguments when you start the container:

    ```shell
      -shm-size=1g -ulimit memlock=-1
    ```

    See also the NCCL [known issue](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data) and the GitHub [issue](https://github.com/NVIDIA-Merlin/HugeCTR/issues/243).
  + `KafkaProducers` startup succeeds even if the target Kafka broker is unresponsive.
    To avoid data loss in conjunction with streaming-model updates from Kafka, you have to make sure that a sufficient number of Kafka brokers are running, operating properly, and are reachable from the node where you run HugeCTR.
  + The number of data files in the file list should be greater than or equal to the number of data reader workers.
    Otherwise, different workers are mapped to the same file and data loading does not progress as expected.
  + Joint loss training with a regularizer is not supported.
  + The Criteo 1 TB click logs dataset that is used with many HugeCTR sample programs and notebooks is currently unavailable.
    Until the dataset becomes downloadable again, you can run those samples based on our synthetic dataset generator.
    For more information, see the [Getting Started](https://github.com/NVIDIA-Merlin/HugeCTR#getting-started) section of the repository README file.


## What's New in Version 3.5
+ **HPS interface encapsulation and exporting as library**: We encapsulate the Hierarchical Parameter Server(HPS) interfaces and deliver it as a standalone library. Besides, we prodvide HPS Python APIs and demonstrate the usage with a notebook. For more information, please refer to [Hierarchical Parameter Server](https://nvidia-merlin.github.io/HugeCTR/v3.5/hugectr_parameter_server.html) and [HPS Demo](notebooks/hps_demo.ipynb).

+ **Hierarchical Parameter Server Triton Backend**: The HPS Backend is a framework for embedding vectors looking up on large-scale embedding tables that was designed to effectively use GPU memory to accelerate the looking up by decoupling the embedding tables and embedding cache from the end-to-end inference pipeline of the deep recommendation model. For more information, please refer to the [samples](https://github.com/triton-inference-server/hugectr_backend/tree/v3.5/samples) directory of the HugeCTR backend for Triton Inference Server repository.

+ **SOK pip release**: SOK pip releases on <https://pypi.org/project/merlin-sok/>. Now users can install SOK via `pip install merlin-sok`.

+ **Joint loss and multi-tasks training support:**: We support joint loss in training so that users can train with multiple labels and tasks with different weights. See the MMoE sample in the [samples/mmoe](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.5/samples/mmoe) directory of the repository to learn the usage.

+ **HugeCTR documentation on web page**: Now users can visit our [web documentation](https://nvidia-merlin.github.io/HugeCTR/main/).

+ **ONNX converter enhancement:**: We enable converting `MultiCrossEntropyLoss` and `CrossEntropyLoss` layers to ONNX to support multi-label inference. For more information, please refer to the HugeCTR to ONNX Converter information in the [onnx_converter](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.5/onnx_converter) directory of the repository.

+ **HDFS python API enhancement**:
    + Simplified `DataSourceParams` so that users do not need to provide all the paths before they are really necessary. Now users only have to pass `DataSourceParams` once when creating a solver.
    + Later paths will be automatically regarded as local paths or HDFS paths depending on the `DataSourceParams` setting. See [notebook](https://nvidia-merlin.github.io/HugeCTR/v3.5/notebooks/training_with_hdfs.html) for usage.

+ **HPS performance optimization**: We use better method to  determine partition number in database backends in HPS.

+ **Issues Fixed**: HugeCTR input layer now can take dense_dim greater than 1000.

## What's New in Version 3.4.1
+ **Support mixed precision inference for dataset with multiple labels**: We enable FP16 for the `Softmax` layer and support mixed precision for multi-label inference. For more information, please refer to [Inference API](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#inference-api).

+ **Support multi-GPU offline inference with Python API**: We support multi-GPU offline inference with the Python interface, which can leverage [Hierarchical Parameter Server](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_parameter_server.html) and enable concurrent execution on multiple devices. For more information, please refer to [Inference API](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#inference-api) and [Multi-GPU Offline Inference Notebook](notebooks/multi_gpu_offline_inference.ipynb).

+ **Introduction to metadata.json**: We add the introduction to `_metadata.json` for Parquet datasets. For more information, please refer to [Parquet](https://nvidia-merlin.github.io/HugeCTR/main/api/index.html).

+ **Documents and tool for workspace size per GPU estimation**: we add a tool that is named the `embedding_workspace_calculator` to help calculate the value for `workspace_size_per_gpu_in_mb` that is required by hugectr.SparseEmbedding. For more information, please refer to the README.md file in the [tools/embedding_workspace_calculator](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.4.1/tools/embedding_workspace_calculator) directory of the repository and [QA 24](https://nvidia-merlin.github.io/HugeCTR/main/QAList.html) in the documentation.

+ **Improved Debugging Capability**: The old logging system, which was flagged as deprecated for some time has been removed. All remaining log messages and outputs have been revised and migrated to the new logging system (base/debug/logging.hpp/cpp). During this revision, we also adjusted log levels for log messages throughout the entire codebase to improve visibility of relevant information.

+ **Support HDFS Parameter Server in Training**:
    + Decoupled HDFS in Merlin containers to make the HDFS support more flexible. Users can now compile HDFS related functionalities optionally.
    + Now supports loading and dumping models and optimizer states from HDFS.
    + Added a [notebook](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.4.1/notebooks/training_with_hdfs.ipynb) to show how to use HugeCTR with HDFS.

+ **Support Multi-hot Inference on Hugectr Backend**: We support categorical input in multi-hot format for HugeCTR Backend inference.

+ **Multi-label inference with mixed precision**: Mixed precision training is enabled for softmax layer.

+ **Python Script and documentation demonstrating how to analyze model files**: In this release, we provide a script to retrieve vocabulary information from model file. Please find more details on the README in the [tools/model_analyzer](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.4.1/tools/model_analyzer) directory of the repository.

+ **Issues fixed**:
    + Mirror strategy bug in SOK (see in https://github.com/NVIDIA-Merlin/HugeCTR/issues/291)
    + Can't import sparse operation kit in nvcr.io/nvidia/merlin/merlin-tensorflow-training:22.04 (see in https://github.com/NVIDIA-Merlin/HugeCTR/issues/296)
    + HPS: Fixed access violation that can occur during initialization when not configuring a volatile DB.



## What's New in Version 3.4

+ **Support for Building HugeCTR with the Unified Merlin Container**: HugeCTR can now be built using our unified Merlin container. For more information, refer to our [Contributor Guide](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_contributor_guide.html).

+ **Hierarchical Parameter Server (HPS) Enhancements**:
    + **New Missing Key (Embedding Table Entries) Insertion Feature**: Using a simple flag, it is now possible to configure HugeCTR with missing keys (embedding table entries). During lookup, these missing keys will automatically be inserted into volatile database layers such as the Redis and Hashmap backends.
    + **Asynchronous Timestamp Refresh**: To allow time-based eviction to take place, it is now possible to enable timestamp refreshing for frequently used embeddings. Once enabled, refreshing is handled asynchronously using background threads, which won’t block your inference jobs. For most applications, the associated performance impact from enabling this feature is barely noticeable.
    + **HDFS (Hadoop Distributed File System) Parameter Server Support During Training**:
        + We're introducing a new DataSourceParams API, which is a python API that can be used to specify the file system and paths to data and model files.
        + We've added support for loading data from HDFS to the local file system for HugeCTR training.
        + We've added support for dumping trained model and optimizer states into HDFS.
    + **New Load API Capabilities**: In addition to being able to deploy new models, the HugeCTR Backend's [Load API](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md#load) can now be used to update the dense parameters for models and corresponding embedding inference cache online.

+ **Sparse Operation Kit (SOK) Enhancements**:
    + **Mixed Precision Training**: Enabling mixed precision training using TensorFlow’s pattern to enhance the training performance and lessen memory usage is now possible.
    + **DLRM Benchmark**: DLRM is a standard benchmark for recommendation model training, so we added a new notebook. Refer to the [sparse_operation_kit/documents/tutorials/DLRM_Benchmark](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.4/sparse_operation_kit/documents/tutorials/DLRM_Benchmark) directory of the repository. The notebook shows how to address the performance of SOK on this benchmark.
    + **Uint32_t / int64_t key dtype Support in SOK**: Int64 or uint32 can now be used as the key data type for SOK’s embedding. Int64 is the default.
    + **TensorFlow Initializers Support**: We now support the TensorFlow native initializer within SOK, such as `sok.All2AllDenseEmbedding(embedding_initializer=tf.keras.initializers.RandomUniform())`. For more information, refer to [All2All Dense Embedding](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/embeddings/dense/all2all.html).

+ **Documentation Enhancements**
    + We've revised several of our notebooks and readme files to improve readability and accessibility.
    + We've revised the SOK docker setup instructions to indicate that HugeCTR setup issues can be resolved using the `--shm-size` setting within docker.
    + Although HugeCTR is designed for scalability, having a robust machine is not necessary for smaller workloads and testing. We've documented the required specifications for notebook testing environments. For more information, refer to our [README for HugeCTR Jupyter Demo Notebooks](https://nvidia-merlin.github.io/HugeCTR/main/notebooks/index.html#system-specifications).

+ **Inference Enhancements**：We now support HugeCTR inference for managing multiple tasks. When the label dimension is the number of binary classification tasks and `MultiCrossEntropyLoss` is employed during training, the shape of inference results will be `(batch_size*num_batches, label_dim)`. For more information, refer to [Inference API](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#inference-api).

+ **Embedding Cache Issue Resolution**: The embedding cache issue for very small embedding tables has been resolved.

## What's New in Version 3.3.1

+ **Hierarchical Parameter Server (HPS) Enhancements**:
    + **HugeCTR Backend Enhancements**: The HugeCTR Backend is now fully compatible with the [Triton model control protocol](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md), so new model configurations can be simply added to the [HPS configuration file](https://github.com/triton-inference-server/hugectr_backend#independent-inference-hierarchical-parameter-server-configuration). The HugeCTR Backend will continue to support online deployments of new models using the Triton Load API. However, with this enhancement, old models can be recycled online using the [Triton Unload API](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md#unload).
    + **Simplified Database Backend**: Multi-nodes, single-node, and all other kinds of volatile database backends can now be configured using the same configuration object.
    + **Multi-Threaded Optimization of Redis Code**: The speedup of HugeCTR version 3.3.1 is 2.3 times faster than HugeCTR version 3.3.
    + **Additional HPS Enhancements and Fixes**:
        + You can now build the HPS test environment and implement unit tests for each component.
        + You'll no longer encounter the access violation issue when updating Apache Kafka online.
    	+ The parquet data reader no longer incorrectly parses the index of categorical features when multiple embedded tables are being used.
    	+ The HPS Redis Backend overflow is now invoked during single insertions.

+ **New GroupDenseLayer**: We're introducing a new GroupDenseLayer. It can be used to group fused fully connected layers when constructing the model graph. A simplified Python interface is provided for adjusting the number of layers and specifying the output dimensions in each layer, which makes it easy to leverage the highly-optimized fused fully connected layers in HugeCTR. For more information, refer to [GroupDenseLayer](https://nvidia-merlin.github.io/HugeCTR/main/api/hugectr_layer_book.html#groupdenselayer).

+ **Global Fixes**:
   + A warning message now appears when attempting to launch a multi-process job before importing the mpi.
   + When running with embedding training cache, a massive log is no longer generated.
   + Legacy conda information has been removed from the HugeCTR documentation.

## What's New in Version 3.3

+ **Hierarchical Parameter Server (HPS) Enhancements**:
    + **Support for Incremental Model Updates**: HPS now supports incremental model updates via Apache Kafka (a distributed event streaming platform) message queues. With this enhancement, HugeCTR can now be connected with Apache Kafka deployments to update models in real time during training and inference. For more information, refer to the [Demo Notebok](https://github.com/triton-inference-server/hugectr_backend/tree/main/samples/hierarchical_deployment/hps_e2e_demo).
    + **Improvements to the Memory Management**: The Redis cluster and CPU memory database backends are now the primary sources for memory management. When performing incremental model updates, these memory database backends will automatically evict infrequently used embeddings as training progresses. The performance of the Redis cluster and CPU memory database backends have also been improved.
    + **New Asynchronous Refresh Mechanism**: Support for asynchronous refreshing of incremental embedding keys into the embedding cache has been added. The Refresh operation will be triggered when completing the model version iteration or outputting incremental parameters from online training. The Distributed Database and Persistent Database will be updated by Apache Kafka. The GPU embedding cache will then refresh the values of the existing embedding keys and replace them with the latest incremental embedding vectors. For more information, refer to the [HPS README](https://github.com/triton-inference-server/hugectr_backend#hugectr-inference-hierarchical-parameter-server).
    + **Configurable Backend Implementations for Databases**: Backend implementations for databases are now fully configurable.
    + **Improvements to the JSON Interface Parser**: The JSON interface parser can now handle inaccurate parameterization.
    + **More Meaningful Jabber**: As requested, we've revised the log levels throughout the entire API database backend of the HPS. Selected configuration options are now printed entirely and uniformly to the log. Errors provide more verbose information about pending issues.

+ **Sparse Operation Kit (SOK) Enhancements**:
    + **TensorFlow (TF) 1.15 Support**: SOK can now be used with TensorFlow 1.15. For more information, refer to [README](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/get_started/get_started.html#tensorflow-1-15).
    + **Dedicated CUDA Stream**: A dedicated CUDA stream is now used for SOK’s Ops, so this may help to eliminate kernel interleaving.
    + **New pip Installation Option**: SOK can now be installed using the `pip install SparseOperationKit` command. See more in our [instructions](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/intro_link.html#installation)). With this install option, root access to compile SOK is no longer required and python scripts don't need to be copied.
    + **Visible Device Configuration Support**：`tf.config.set_visible_device` can now be used to set visible GPUs for each process. `CUDA_VISIBLE_DEVICES` can also be used. When `tf.distribute.Strategy` is used, the `tf.config.set_visible_device` argument shouldn't be set.
+ **Hybrid-embedding indices pre-computing**：The indices needed for hybrid embedding are pre-computed ahead of time and are overlapped with previous iterations.

+ **Cached evaluation indices:**：The hybrid-embedding indices for eval are cached when applicable, hence eliminating the re-computing of the indices at every eval iteration.

+ **MLP weight/data gradients calculation overlap:**：The weight gradients of MLP are calculated asynchronously with respect to the data gradients, enabling overlap between these two computations.

+ **Better compute-communication overlap:**：Better overlap between compute and communication has been enabled to improve training throughput.

+ **Fused weight conversion:**：The FP32-to-FP16 conversion of the weights are now fused into the SGD optimizer, saving trips to memory.

+ **GraphScheduler:**：GrapScheduler was added to control the timing of cudaGraph launching. With GraphScheduler, the gap between adjacent cudaGraphs is eliminated.

+ **Multi-Node Training Support Enhancements**：You can now perform multi-node training on the cluster with non-RDMA hardware by setting the `AllReduceAlgo.NCCL` value for the `all_reduce_algo` argument. For more information, refer to the details for the `all_reduce_algo` argument in the [CreateSolver API](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#solver).

+ **Support for Model Naming During Model Dumping**: You can now specify names for models with the `CreateSolver`training API, which will be dumped to the JSON configuration file with the `Model.graph_to_json` API. This will facilitate the Triton deployment of saved HugeCTR models, as well as help to distinguish between models when Apache Kafka sends parameters from training to inference.

+ **Fine-Grained Control Accessibility Enhancements for Embedding Layers**: We've added fine-grained control accessibility to embedding layers. Using the `Model.freeze_embedding` and `Model.unfreeze_embedding` APIs, embedding layer weights can be frozen and unfrozen. Additionally, weights for multiple embedding layers can be loaded independently, making it possible to load pre-trained embeddings for a particular layer. For more information, refer to [Model API](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#model) and [Section 3.4 of the HugeCTR Criteo Notebook](https://github.com/NVIDIA-Merlin/HugeCTR/blob/v3.3/notebooks/hugectr_criteo.ipynb).

## What's New in Version 3.2.1

+ **GPU Embedding Cache Optimization**: The performance of the GPU embedding cache for the standalone module has been optimized. With this enhancement, the performance of small to medium batch sizes has improved significantly. We're not introducing any changes to the interface for the GPU embedding cache, so don't worry about making changes to any existing code that uses this standalone module. For more information, refer to the `ReadMe.md` file in the [gpu_cache](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.2.1/gpu_cache) directory of the repository.

+ **Model Oversubscription Enhancements**: We're introducing a new host memory cache (HMEM-Cache) component for the model oversubscription feature. When configured properly, incremental training can be efficiently performed on models with large embedding tables that exceed the host memory. For more information, refer to [Host Memory Cache in MOS](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_core_features.html#embedding-training-cache). Additionally, we've enhanced the Python interface for model oversubscription by replacing the `use_host_memory_ps` parameter with a `ps_types` parameter and adding a `sparse_models` parameter. For more information about these changes, refer to [HugeCTR Python Interface](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#solver).

+ **Debugging Enhancements**: We're introducing new debugging features such as multi-level logging, as well as kernel debugging functions. We're also making our error messages more informative so that users know exactly how to resolve issues related to their training and inference code. For more information, refer to the comments in the header files, which are available at HugeCTR/include/base/debug.

+ **Enhancements to the Embedding Key Insertion Mechanism for the Embedding Cache**: Missing embedding keys can now be asynchronously inserted into the embedding cache. To enable automatically, set the hit rate threshold within the configuration file. When the actual hit rate of the embedding cache is higher than the hit rate threshold that the user set or vice versa, the embedding cache will insert the missing embedding key asynchronously.

+ **Parameter Server Enhancements**: We're introducing a new "in memory" database that utilizes the local CPU memory for storing and recalling embeddings and uses multi-threading to accelerate lookup and storage. You can now also use the combined CPU-accessible memory of your Redis cluster to store embeddings. We improved the performance for the "persistent" storage and retrieving embeddings from RocksDB using structured column families, as well as added support for creating hierarchical storage such as Redis as distributed cache. You don't have to worry about updating your Parameter Server configurations to take advantage of these enhancements.

+ **Slice Layer Internalization Enhancements**: The Slice layer for the branch toplogy can now be abstracted away in the Python interface. A model graph analysis will be conducted to resolve the tensor dependency and the Slice layer will be internally inserted if the same tensor is consumed more than once to form the branch topology. For more information about how to construct a model graph using branches without the Slice layer, refer to the Getting Started section of the repository README and the [Slice Layer](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#layers) information.

## What's New in Version 3.2

+ **New HugeCTR to ONNX Converter**: We’re introducing a new HugeCTR to ONNX converter in the form of a Python package. All graph configuration files are required and model weights must be formatted as inputs. You can specify where you want to save the converted ONNX model. You can also convert sparse embedding models. For more information, refer to the HugeCTR to ONNX Converter information in the [onnx_converter](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.2/onnx_converter) directory and the [HugeCTR2ONNX Demo Notebook](notebooks/hugectr2onnx_demo.ipynb).

+ **New Hierarchical Storage Mechanism on the Parameter Server (POC)**: We’ve implemented a hierarchical storage mechanism between local SSDs and CPU memory. As a result, embedding tables no longer have to be stored in the local CPU memory. The distributed Redis cluster is being implemented as a CPU cache to store larger embedding tables and interact with the GPU embedding cache directly. The local RocksDB serves as a query engine to back up the complete embedding table on the local SSDs and assist the Redis cluster with looking up missing embedding keys. For more information about how this works, refer to our [HugeCTR Backend documentation](https://github.com/triton-inference-server/hugectr_backend/blob/main/docs/architecture.md#distributed-deployment-with-hierarchical-hugectr-parameter-server)

+ **Parquet Format Support within the Data Generator**: The HugeCTR data generator now supports the parquet format, which can be configured easily using the Python API. For more information, refer to [Data Generator API](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#data-generator-api).

+ **Python Interface Support for the Data Generator**: The data generator has been enabled within the HugeCTR Python interface. The parameters associated with the data generator have been encapsulated into the `DataGeneratorParams` struct, which is required to initialize the `DataGenerator` instance. You can use the data generator's Python APIs to easily generate the Norm, Parquet, or Raw dataset formats with the desired distribution of sparse keys. For more information, refer to [Data Generator API](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#data-generator-api) and the data generator samples in the [tools/data_generator](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.2/tools/data_generator) directory of the repository.

+ **Improvements to the Formula of the Power Law Simulator within the Data Generator**: We've modified the formula of the power law simulator within the data generator so that a positive alpha value is always produced, which will be needed for most use cases. The alpha values for `Long`, `Medium`, and `Short` within the power law distribution are 0.9, 1.1, and 1.3 respectively. For more information, refer to [Data Generator API](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#data-generator-api).

+ **Support for Arbitrary Input and Output Tensors in the Concat and Slice Layers**: The Concat and Slice layers now support any number of input and output tensors. Previously, these layers were limited to a maximum of four tensors.

+ **New Continuous Training Notebook**: We’ve added a new notebook to demonstrate how to perform continuous training using the embedding training cache (also referred to as Embedding Training Cache) feature. For more information, refer to [HugeCTR Continuous Training](notebooks/continuous_training.ipynb).

+ **New HugeCTR Contributor Guide**: We've added a new [HugeCTR Contributor Guide](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_contributor_guide.html) that explains how to contribute to HugeCTR, which may involve reporting and fixing a bug, introducing a new feature, or implementing a new or pending feature.

+ **Sparse Operation Kit (SOK) Enhancements**: SOK now supports TensorFlow 2.5 and 2.6. We also added support for identity hashing, dynamic input, and Horovod within SOK. Lastly, we added a new [SOK docs set](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/index.html) to help you get started with SOK.

## What's New in Version 3.1

+ **Hybrid Embedding**: Hybrid embedding is designed to overcome the bandwidth constraint imposed by the embedding part of the embedding train workload by algorithmically reducing the traffic over netwoek. Requirements: The input dataset has only one-hot feature items and the model uses the SGD optimizer.

+ **FusedReluBiasFullyConnectedLayer**: FusedReluBiasFullyConnectedLayer is one of the major optimizations applied to dense layers. It fuses relu Bias and FullyConnected layers to reduce the memory access on HBM. Requirements: The model uses a layer with separate data / gradient tensors as the bottom layer.

+ **Overlapped Pipeline**: The computation in the dense input data path is overlapped with the hybrid embedding computation. Requirements: The data reader is asynchronous, hybrid embedding is used, and the model has a feature interaction layer.

+ **Holistic CUDA Graph**: Packing everything inside a training iteration into a CUDA Graph. Limitations: this option works only if use_cuda_graph is turned off and use_overlapped_pipeline is turned on.

+ **Python Interface Enhancements**: We’ve enhanced the Python interface for HugeCTR so that you no longer have to manually create a JSON configuration file. Our Python APIs can now be used to create the computation graph. They can also be used to dump the model graph as a JSON object and save the model weights as binary files so that continuous training and inference can take place. We've added an Inference API that takes Norm or Parquet datasets as input to facilitate the inference process. For more information, refer to [HugeCTR Python Interface](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html) and [HugeCTR Criteo Notebook](notebooks/hugectr_criteo.ipynb).

+ **New Interface for Unified Embedding**: We’re introducing a new interface to simplify the use of embeddings and datareaders. To help you specify the number of keys in each slot, we added `nnz_per_slot` and `is_fixed_length`. You can now directly configure how much memory usage you need by specifying `workspace_size_per_gpu_in_mb` instead of `max_vocabulary_size_per_gpu`. For convenience, `mean/sum` is used in combinators instead of 0 and 1. In cases where you don't know which embedding type you should use, you can specify `use_hash_table` and let HugeCTR automatically select the embedding type based on your configuration. For more information, refer to [HugeCTR Python Interface](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html).

+ **Multi-Node Support for Embedding Training Cache (ETC)**: We’ve enabled multi-node support for the embedding training cache. You can now train a model with a terabyte-size embedding table using one node or multiple nodes even if the entire embedding table can't fit into the GPU memory. We're also introducing the host memory (HMEM) based parameter server (PS) along with its SSD-based counterpart. If the sparse model can fit into the host memory of each training node, the optimized HMEM-based PS can provide better model loading and dumping performance with a more effective bandwidth. For more information, refer to [HugeCTR Python Interface](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html).

+ **Enhancements to the Multi-Nodes TensorFlow Plugin**: The Multi-Nodes TensorFlow Plugin now supports multi-node synchronized training via tf.distribute.MultiWorkerMirroredStrategy. With minimal code changes, you can now easily scale your single GPU training to multi-node multi GPU training. The Multi-Nodes TensorFlow Plugin also supports multi-node synchronized training via Horovod. The inputs for embedding plugins are now data parallel, so the datareader no longer needs to preprocess data for different GPUs based on concrete embedding algorithms. For more information, see the `sparse_operation_kit_demo.ipynb` notebook in the [sparse_operation_kit/notebooks](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.2/sparse_operation_kit/notebooks) directory of the repository.

+ **NCF Model Support**: We've added support for the NCF model, as well as the GMF and NeuMF variant models. With this enhancement, we're introducing a new element-wise multiplication layer and HitRate evaluation metric. Sample code was added that demonstrates how to preprocess user-item interaction data and train a NCF model with it. New examples have also been added that demonstrate how to train NCF models using MovieLens datasets.

+ **DIN and DIEN Model Support**: All of our layers support the DIN model. The following layers support the DIEN model: FusedReshapeConcat, FusedReshapeConcatGeneral, Gather, GRU, PReLUDice, ReduceMean, Scale, Softmax, and Sub. We also added sample code to demonstrate how to use the Amazon dataset to train the DIN model. See our sample programs in the [samples/din](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.1/samples/din) directory of the repository.

+ **Multi-Hot Support for Parquet Datasets**: We've added multi-hot support for parquet datasets, so you can now train models with a paraquet dataset that contains both one hot and multi-hot slots.

+ **Mixed Precision (FP16) Support in More Layers**: The MultiCross layer now supports mixed precision (FP16). All layers now support FP16.

+ **Mixed Precision (FP16) Support in Inference**: We've added FP16 support for the inference pipeline. Therefore, dense layers can now adopt FP16 during inference.

+ **Optimizer State Enhancements for Continuous Training**: You can now store optimizer states that are updated during continuous training as files, such as the Adam optimizer's first moment (m) and second moment (v). By default, the optimizer states are initialized with zeros, but you can specify a set of optimizer state files to recover their previous values. For more information about `dense_opt_states_file` and `sparse_opt_states_file`, refer to [Python Interface](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#).

+ **New Library File for GPU Embedding Cache Data**: We’ve moved the header/source code of the GPU embedding cache data structure into a stand-alone folder. It has been compiled into a stand-alone library file. Similar to HugeCTR, your application programs can now be directly linked from this new library file for future use. For more information, refer to the `ReadMe.md` file in the [gpu_cache](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.1/gpu_cache) directory of the repository.

+ **Embedding Plugin Enhancements**: We’ve moved all the embedding plugin files into a stand-alone folder. The embedding plugin can be used as a stand-alone python module, and works with TensorFlow to accelerate the embedding training process.

+ **Adagrad Support**: Adagrad can now be used to optimize your embedding and network. To use it, change the optimizer type in the Optimizer layer and set the corresponding parameters.

## What's New in Version 3.0.1

+ **New DLRM Inference Benchmark**: We've added two detailed Jupyter notebooks to demonstrate how to train, deploy, and benchmark the performance of a deep learning recommendation model (DLRM) with HugeCTR. For more information, refer to our [HugeCTR Inference Notebooks](https://github.com/triton-inference-server/hugectr_backend/tree/v3.0.1/samples/dlrm).

+ **FP16 Optimization**: We've optimized the DotProduct, ELU, and Sigmoid layers based on `__half2` vectorized loads and stores, improving their device memory bandwidth utilization. MultiCross, FmOrder2, ReduceSum, and Multiply are the only layers that still need to be optimized for FP16.

+ **Synthetic Data Generator Enhancements**: We've enhanced our synthetic data generator so that it can generate uniformly distributed datasets, as well as power-law based datasets. You can now specify the `vocabulary_size` and `max_nnz` per categorical feature instead of across all categorial features. For more information, refer to our [user guide](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_user_guide.html#generating-synthetic-data-and-benchmarks).

+ **Reduced Memory Allocation for Trained Model Exportation**: To prevent the "Out of Memory" error message from displaying when exporting a trained model, which may include a very large embedding table, the amount of memory allocated by the related functions has been significantly reduced.

+ **Dropout Layer Enhancement**: The Dropout layer is now compatible with CUDA Graph. The Dropout layer is using cuDNN by default so that it can be used with CUDA Graph.

## What’s New in Version 3.0

+ **Inference Support**: To streamline the recommender system workflow, we’ve implemented a custom HugeCTR backend on the [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server). The HugeCTR backend leverages the embedding cache and parameter server to efficiently manage embeddings of different sizes and models in a hierarchical manner. For more information, refer to [our inference repository](https://github.com/triton-inference-server/hugectr_backend).

+ **New High-Level API**: You can now also construct and train your models using the Python interface with our new high-level API. For more information, refer to our preview example code in the `samples/preview` directory to grasp how this new API works.

+ **[FP16 Support](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_core_features.html#mixed-precision-training) in More Layers**: All the layers except `MultiCross` support mixed precision mode. We’ve also optimized some of the FP16 layer implementations based on vectorized loads and stores.

+ **Enhanced TensorFlow Embedding Plugin**: Our embedding plugin now supports `LocalizedSlotSparseEmbeddingHash` mode. With this enhancement, the DNN model no longer needs to be split into two parts since it now connects with the embedding op through `MirroredStrategy` within the embedding layer. For more information, see the `notebooks/embedding_plugin.ipynb` notebook.

+ **Extended Embedding Training Cache**: We’ve extended the embedding training cache feature to support `LocalizedSlotSparseEmbeddingHash` and `LocalizedSlotSparseEmbeddingHashOneHot`.

+ **Epoch-Based Training Enhancements**: The `num_epochs` option in the **Solver** clause can now be used with the `Raw` dataset format.

+ **Deprecation of the `eval_batches` Parameter**: The `eval_batches` parameter has been deprecated and replaced with the `max_eval_batches` and `max_eval_samples` parameters. In epoch mode, these parameters control the maximum number of evaluations. An error message will appear when attempting to use the `eval_batches` parameter.

+ **`MultiplyLayer` Renamed**: To clarify what the `MultiplyLayer` does, it was renamed to `WeightMultiplyLayer`.

+ **Optimized Initialization Time**: HugeCTR’s initialization time, which includes the GEMM algorithm search and parameter initialization, was significantly reduced.

+ **Sample Enhancements**: Our samples now rely upon the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) instead of the Kaggle Display Advertising Challenge dataset. Our preprocessing scripts (Perl, Pandas, and NVTabular) have also been unified and simplified.

+ **Configurable DataReader Worker**: You can now specify the number of data reader workers, which run in parallel, with the `num_workers` parameter. Its default value is 12. However, if you are using the Parquet data reader, you can't configure the `num_workers` parameter since it always corresponds to the number of active GPUs.

## What's New in Version 2.3

+ **New Python Interface**: To enhance the interoperability with [NVTabular](https://github.com/NVIDIA/NVTabular) and other Python-based libraries, we're introducing a new Python interface for HugeCTR.

+ **HugeCTR Embedding with Tensorflow**: To help users easily integrate HugeCTR’s optimized embedding into their Tensorflow workflow, we now offer the HugeCTR embedding layer as a Tensorflow plugin. To better understand how to install, use, and verify it, see our Jupyter notebook tutorial in file `notebooks/embedding_plugin.ipynb`. The notebook also demonstrates how you can create a new Keras layer, `EmbeddingLayer`, based on the `hugectr.py` file in the `tools/embedding_plugin/python` directory with the helper code that we provide.

+ **Embedding Training Cache**: To enable a model with large embedding tables that exceeds the single GPU's memory limit, we've added a new embedding training cache feature, giving you the ability to load a subset of an embedding table into the GPU in a coarse grained, on-demand manner during the training stage.

+ **TF32 Support**: We've added TensorFloat-32 (TF32), a new math mode and third-generation of Tensor Cores, support on Ampere. TF32 uses the same 10-bit mantissa as FP16 to ensure accuracy while providing the same range as FP32 by using an 8-bit exponent. Since TF32 is an internal data type that accelerates FP32 GEMM computations with tensor cores, you can simply turn it on with a newly added configuration option. For more information, refer to [Solver](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#solver).

+ **Enhanced AUC Implementation**: To enhance the performance of our AUC computation on multi-node environments, we've redesigned our AUC implementation to improve how the computational load gets distributed across nodes.

+ **Epoch-Based Training**: In addition to the `max_iter` parameter, you can now set the `num_epochs` parameter in the **Solver** clause within the configuration file. This mode can only currently be used with `Norm` dataset formats and their corresponding file lists. All dataset formats will be supported in the future.

+ **New Multi-Node Training Tutorial**: To better support multi-node training use cases, we've added a new step-by-step tutorial to the [tutorial/multinode-training](https://github.com/NVIDIA-Merlin/HugeCTR/tree/main/tutorial/multinode-training) directory of our GitHub repository.

+ **Power Law Distribution Support with Data Generator**: Because of the increased need for generating a random dataset whose categorical features follows the power-law distribution, we've revised our data generation tool to support this use case. For additional information, refer to the `--long-tail` description in the Generating Synthetic Data and Benchmarks section of the [docs/hugectr_user_guide.md](https://github.com/NVIDIA-Merlin/HugeCTR/blob/v2.3/docs/hugectr_user_guide.md#generating-synthetic-data-and-benchmarks) file in the repository.

+ **Multi-GPU Preprocessing Script for Criteo Samples**: Multiple GPUs can now be used when preparing the dataset for the programs in the [samples](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v2.3/samples) directory of our GitHub repository. For more information, see how the `preprocess_nvt.py` program in the [tools/criteo_script](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v2.3/tools/criteo_script) directory of the repository is used to preprocess the Criteo dataset for DCN, DeepFM, and W&D samples.

## Known Issues

+ HugeCTR uses NCCL to share data between ranks, and NCCL may require shared system memory for IPC and pinned (page-locked) system memory resources. When using NCCL inside a container, it is recommended that you increase these resources by issuing: `-shm-size=1g -ulimit memlock=-1`
See also [NCCL's known issue](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data). And the [GitHub issue](https://github.com/NVIDIA-Merlin/HugeCTR/issues/243).

+ KafkaProducers startup will succeed, even if the target Kafka broker is unresponsive. In order to avoid data-loss in conjunction with streaming model updates from Kafka, you have to make sure that a sufficient number of Kafka brokers is up, working properly and reachable from the node where you run HugeCTR.

+ The number of data files in the file list should be no less than the number of data reader workers. Otherwise, different workers will be mapped to the same file and data loading does not progress as expected.

+ Joint Loss training hasn’t been supported with regularizer.

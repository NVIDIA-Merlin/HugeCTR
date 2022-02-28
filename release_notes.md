# Release Notes

## What's New in Version 3.4

+ **Support for Building HugeCTR with the Unified Merlin Container**: HugeCTR can now be built using our unified Merlin container. For more information, refer to our [Contributor Guide](docs/hugectr_contributor_guide.md).

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
    + **DLRM Benchmark**: DLRM is a standard benchmark for recommendation model training, so we added a new [notebook](sparse_operation_kit/documents/tutorials/DLRM_Benchmark#sok-dlrm-benchmark) to address the performance of SOK on this benchmark.
    + **Uint32_t / int64_t key dtype Support in SOK**: Int64 or uint32 can now be used as the key data type for SOK’s embedding. Int64 is the default.
    + **TensorFlow Initializers Support**: We now support the TensorFlow native initializer within SOK, such as `sok.All2AllDenseEmbedding(embedding_initializer=tf.keras.initializers.RandomUniform())`. For more information, refer to [All2All Dense Embedding](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/embeddings/dense/all2all.html).

+ **Documentation Enhancements**
    + We've revised several of our notebooks and readme files to improve readability and accessibility.
    + We've revised the SOK docker setup instructions to indicate that HugeCTR setup issues can be resolved using the `--shm-size` setting within docker.
    + Although HugeCTR is designed for scalability, having a robust machine is not necessary for smaller workloads and testing. We've documented the required specifications for notebook testing environments. For more information, refer to our [README for HugeCTR Jupyter Demo Notebooks]( /notebooks#system-specifications).

+ **Inference Enhancements**：We now support HugeCTR inference for managing multiple tasks. When the label dimension is the number of binary classification tasks and `MultiCrossEntropyLoss` is employed during training, the shape of inference results will be `(batch_size*num_batches, label_dim)`. For more information, refer to [Inference API](docs/python_interface.md#predict-method).

+ **Embedding Cache Issue Resolution**: The embedding cache issue for very small embedding tables has been resolved.

## What's New in Version 3.3.1

+ **Hierarchical Parameter Server (HPS) Enhancements**:
    + **HugeCTR Backend Enhancements**: The HugeCTR Backend is now fully compatible with the [Triton model control protocol](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md), so new model configurations can be simply added to the [HPS configuration file](https://github.com/triton-inference-server/hugectr_backend#independent-parameter-server-configuration). The HugeCTR Backend will continue to support online deployments of new models using the Triton Load API. However, with this enhancement, old models can be recycled online using the [Triton Unload API](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md#unload).
    + **Simplified Database Backend**: Multi-nodes, single-node, and all other kinds of volatile database backends can now be configured using the same configuration object.
    + **Multi-Threaded Optimization of Redis Code**: The speedup of HugeCTR version 3.3.1 is 2.3 times faster than HugeCTR version 3.3.
    + **Additional HPS Enhancements and Fixes**:
        + You can now build the HPS test environment and implement unit tests for each component.
        + You'll no longer encounter the access violation issue when updating Apache Kafka online.
    	+ The parquet data reader no longer incorrectly parses the index of categorical features when multiple embedded tables are being used.
    	+ The HPS Redis Backend overflow is now invoked during single insertions.
    
+ **New GroupDenseLayer**: We're introducing a new GroupDenseLayer. It can be used to group fused fully connected layers when constructing the model graph. A simplified Python interface is provided for adjusting the number of layers and specifying the output dimensions in each layer, which makes it easy to leverage the highly-optimized fused fully connected layers in HugeCTR. For more information, refer to [GroupDenseLayer](docs/python_interface.md#groupdenselayer).

+ **Global Fixes**: 
   + A warning message now appears when attempting to launch a multi-process job before importing the mpi.
   + When running with embedding training cache, a massive log is no longer generated.
   + Legacy conda information has been removed from the HugeCTR documentation.

## What's New in Version 3.3

+ **Hierarchical Parameter Server (HPS) Enhancements**: 
    + **Support for Incremental Model Updates**: HPS now supports incremental model updates via Apache Kafka (a distributed event streaming platform) message queues. With this enhancement, HugeCTR can now be connected with Apache Kafka deployments to update models in real time during training and inference. For more information, refer to the [Demo Notebok](https://github.com/triton-inference-server/hugectr_backend/tree/main/samples/hierarchical_deployment/hps_e2e_demo).
    + **Improvements to the Memory Management**: The Redis cluster and CPU memory database backends are now the primary sources for memory management. When performing incremental model updates, these memory database backends will automatically evict infrequently used embeddings as training progresses. The performance of the Redis cluster and CPU memory database backends have also been improved.
    + **New Asynchronous Refresh Mechanism**: Support for asynchronous refreshing of incremental embedding keys into the embedding cache has been added. The Refresh operation will be triggered when completing the model version iteration or outputting incremental parameters from online training. The Distributed Database and Persistent Database will be updated by Apache Kafka. The GPU embedding cache will then refresh the values of the existing embedding keys and replace them with the latest incremental embedding vectors. For more information, refer to the [HPS README](https://github.com/triton-inference-server/hugectr_backend#hugectr-hierarchical-parameter-server).
    + **Configurable Backend Implementations for Databases**: Backend implementations for databases are now fully configurable. 
    + **Improvements to the JSON Interface Parser**: The JSON interface parser can now handle inaccurate parameterization.
    + **More Meaningful Jabber**: As requested, we've revised the log levels throughout the entire API database backend of the HPS. Selected configuration options are now printed entirely and uniformly to the log. Errors provide more verbose information about pending issues. 

+ **Sparse Operation Kit (SOK) Enhancements**:
    + **TensorFlow (TF) 1.15 Support**: SOK can now be used with TensorFlow 1.15. For more information, refer to [README](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/get_started/get_started.html#tensorflow-1-15).
    + **Dedicated CUDA Stream**: A dedicated CUDA stream is now used for SOK’s Ops, so this may help to eliminate kernel interleaving.
    + **New pip Installation Option**: SOK can now be installed using the pip install SparseOperationKit (see more in our [instructions](sparse_operation_kit/ReadMe.md#install-this-module-from-pypi)). With this install option, root access to compile SOK is no longer required and python scripts don't need to be copied.
    + **Visible Device Configuration Support**：`tf.config.set_visible_device` can now be used to set visible GPUs for each process. `CUDA_VISIBLE_DEVICES` can also be used. When `tf.distribute.Strategy` is used, the `tf.config.set_visible_device` argument shouldn't be set.
    + **Hanging Issue Fix**: There was a hanging issue in `tf.distribute.MirroredStrategy` when using TensorFlow version 2.4 and higher. However, this is no longer an issue when using TensorFlow version 2.5 and higher.

+ **MLPerf v1.1 Integration Enhancements**：
    + **Precomputed Hybrid Embedding Indices**：The necessary indices for hybrid embedding are now precomputed ahead of time and overlapped with previous iterations.
    + **Cached Eval Indices:**：The hybrid embedding indices for eval are cached when applicable. Index re-computing is no longer needed at every eval iteration.
    + **MLP Weight/Data Gradients Calculation Overlap:**：The weight gradients of MLP are calculated asynchronously with respect to the data gradients, enabling overlap between these two computations.
    + **Improved Compute/Communication Overlap:**：Enhancements to the overlap between the compute and communication has been implemented to improve training throughput.
    + **Fused Weight Conversion:**：The FP32-to-FP16 conversion of the weights are now fused into the SGD optimizer, saving trips to memory.
    + **GraphScheduler Support:**：GrapScheduler was added to control the cudaGraph launch timing. With GraphScheduler, the gap between adjacent cudaGraphs has been eliminated.

+ **Multi-Node Training Support Enhancements**：You can now perform multi-node training on the cluster with non-RDMA hardware by setting the `AllReduceAlgo.NCCL` value for the `all_reduce_algo` argument. For more information, refer to the details for the `all_reduce_algo` argument in the [CreateSolver API](docs/python_interface.md#createsolver-method).

+ **Support for Model Naming During Model Dumping**: You can now specify names for models with the `CreateSolver`training API, which will be dumped to the JSON configuration file with the `Model.graph_to_json` API. This will facilitate the Triton deployment of saved HugeCTR models, as well as help to distinguish between models when Apache Kafka sends parameters from training to inference.

+ **Fine-Grained Control Accessibility Enhancements for Embedding Layers**: We've added fine-grained control accessibility to embedding layers. Using the `Model.freeze_embedding` and `Model.unfreeze_embedding` APIs, embedding layer weights can be frozen and unfrozen. Additionally, weights for multiple embedding layers can be loaded independently, making it possible to load pre-trained embeddings for a particular layer. For more information, refer to [Model API](docs/python_interface.md#model) and [Section 3.4 of the HugeCTR Criteo Notebook](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/notebooks/hugectr_criteo.ipynb).

## What's New in Version 3.2.1

+ **GPU Embedding Cache Optimization**: The performance of the GPU embedding cache for the standalone module has been optimized. With this enhancement, the performance of small to medium batch sizes has improved significantly. We're not introducing any changes to the interface for the GPU embedding cache, so don't worry about making changes to any existing code that uses this standalone module. For more information, refer to the [GPU Cache ReadMe](./gpu_cache/ReadMe.md).

+ **Model Oversubscription Enhancements**: We're introducing a new host memory cache (HMEM-Cache) component for the model oversubscription feature. When configured properly, incremental training can be efficiently performed on models with large embedding tables that exceed the host memory. For more information, refer to [Host Memory Cache in MOS](./docs/intro_hmem_cache.md). Additionally, we've enhanced the Python interface for model oversubscription by replacing the `use_host_memory_ps` parameter with a `ps_types` parameter and adding a `sparse_models` parameter. For more information about these changes, refer to [HugeCTR Python Interface](./docs/python_interface.md#createmos-method).

+ **Debugging Enhancements**: We're introducing new debugging features such as multi-level logging, as well as kernel debugging functions. We're also making our error messages more informative so that users know exactly how to resolve issues related to their training and inference code. For more information, refer to the comments in the header files, which are available at HugeCTR/include/base/debug.

+ **Enhancements to the Embedding Key Insertion Mechanism for the Embedding Cache**: Missing embedding keys can now be asynchronously inserted into the embedding cache. To enable automatically, set the hit rate threshold within the configuration file. When the actual hit rate of the embedding cache is higher than the hit rate threshold that the user set or vice versa, the embedding cache will insert the missing embedding key asynchronously.

+ **Parameter Server Enhancements**: We're introducing a new "in memory" database that utilizes the local CPU memory for storing and recalling embeddings and uses multi-threading to accelerate lookup and storage. You can now also use the combined CPU-accessible memory of your Redis cluster to store embeddings. We improved the performance for the "persistent" storage and retrieving embeddings from RocksDB using structured column families, as well as added support for creating hierarchical storage such as Redis as distributed cache. You don't have to worry about updating your Parameter Server configurations to take advantage of these enhancements.

+ **Slice Layer Internalization Enhancements**: The Slice layer for the branch toplogy can now be abstracted away in the Python interface. A model graph analysis will be conducted to resolve the tensor dependency and the Slice layer will be internally inserted if the same tensor is consumed more than once to form the branch topology. For more information about how to construct a model graph using branches without the Slice layer, refer to [Getting Started](README.md#getting-started) and [Slice Layer](./docs/hugectr_layer_book.md#slice-layer).

## What's New in Version 3.2

+ **New HugeCTR to ONNX Converter**: We’re introducing a new HugeCTR to ONNX converter in the form of a Python package. All graph configuration files are required and model weights must be formatted as inputs. You can specify where you want to save the converted ONNX model. You can also convert sparse embedding models. For more information, refer to [HugeCTR to ONNX Converter](./onnx_converter) and [HugeCTR2ONNX Demo Notebook](notebooks/hugectr2onnx_demo.ipynb).

+ **New Hierarchical Storage Mechanism on the Parameter Server (POC)**: We’ve implemented a hierarchical storage mechanism between local SSDs and CPU memory. As a result, embedding tables no longer have to be stored in the local CPU memory. The distributed Redis cluster is being implemented as a CPU cache to store larger embedding tables and interact with the GPU embedding cache directly. The local RocksDB serves as a query engine to back up the complete embedding table on the local SSDs and assist the Redis cluster with looking up missing embedding keys. For more information about how this works, refer to our [HugeCTR Backend documentation](https://github.com/triton-inference-server/hugectr_backend/blob/main/docs/architecture.md#distributed-deployment-with-hierarchical-hugectr-parameter-server)

+ **Parquet Format Support within the Data Generator**: The HugeCTR data generator now supports the parquet format, which can be configured easily using the Python API. For more information, refer to [Data Generator API](docs/python_interface.md#data-generator-api).

+ **Python Interface Support for the Data Generator**: The data generator has been enabled within the HugeCTR Python interface. The parameters associated with the data generator have been encapsulated into the `DataGeneratorParams` struct, which is required to initialize the `DataGenerator` instance. You can use the data generator's Python APIs to easily generate the Norm, Parquet, or Raw dataset formats with the desired distribution of sparse keys. For more information, refer to [Data Generator API](docs/python_interface.md#data-generator-api) and [Data Generator Samples](tools/data_generator).

+ **Improvements to the Formula of the Power Law Simulator within the Data Generator**: We've modified the formula of the power law simulator within the data generator so that a positive alpha value is always produced, which will be needed for most use cases. The alpha values for `Long`, `Medium`, and `Short` within the power law distribution are 0.9, 1.1, and 1.3 respectively. For more information, refer to [Data Generator API](docs/python_interface.md#data-generator-api).

+ **Support for Arbitrary Input and Output Tensors in the Concat and Slice Layers**: The Concat and Slice layers now support any number of input and output tensors. Previously, these layers were limited to a maximum of four tensors.

+ **New Continuous Training Notebook**: We’ve added a new notebook to demonstrate how to perform continuous training using the embedding training cache (also referred to as Embedding Training Cache) feature. For more information, refer to [HugeCTR Continuous Training](notebooks/continuous_training.ipynb).

+ **New HugeCTR Contributor Guide**: We've added a new [HugeCTR Contributor Guide](docs/hugectr_contributor_guide.md) that explains how to contribute to HugeCTR, which may involve reporting and fixing a bug, introducing a new feature, or implementing a new or pending feature.

+ **Sparse Operation Kit (SOK) Enhancements**: SOK now supports TensorFlow 2.5 and 2.6. We also added support for identity hashing, dynamic input, and Horovod within SOK. Lastly, we added a new [SOK docs set](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/v1.0.0/index.html) to help you get started with SOK.

## What's New in Version 3.1

+ **MLPerf v1.0 Integration**: We've integrated MLPerf optimizations for DLRM training and enabled them as configurable options in Python interface. Specifically, we have incorporated AsyncRaw data reader, HybridEmbedding, FusedReluBiasFullyConnectedLayer, overlapped pipeline, holistic CUDA Graph and so on. The performance of 14-node DGX-A100 DLRM training with Python APIs is comparable to CLI usage. For more information, refer to [HugeCTR Python Interface](docs/python_interface.md) and [DLRM Sample](samples/dlrm).

+ **Python Interface Enhancements**: We’ve enhanced the Python interface for HugeCTR so that you no longer have to manually create a JSON configuration file. Our Python APIs can now be used to create the computation graph. They can also be used to dump the model graph as a JSON object and save the model weights as binary files so that continuous training and inference can take place. We've added an Inference API that takes Norm or Parquet datasets as input to facilitate the inference process. For more information, refer to [HugeCTR Python Interface](docs/python_interface.md) and [HugeCTR Criteo Notebook](notebooks/hugectr_criteo.ipynb).

+ **New Interface for Unified Embedding**: We’re introducing a new interface to simplify the use of embeddings and datareaders. To help you specify the number of keys in each slot, we added `nnz_per_slot` and `is_fixed_length`. You can now directly configure how much memory usage you need by specifying `workspace_size_per_gpu_in_mb` instead of `max_vocabulary_size_per_gpu`. For convenience, `mean/sum` is used in combinators instead of 0 and 1. In cases where you don't know which embedding type you should use, you can specify `use_hash_table` and let HugeCTR automatically select the embedding type based on your configuration. For more information, refer to [HugeCTR Python Interface](docs/python_interface.md).

+ **Multi-Node Support for Embedding Training Cache (ETC)**: We’ve enabled multi-node support for the embedding training cache. You can now train a model with a terabyte-size embedding table using one node or multiple nodes even if the entire embedding table can't fit into the GPU memory. We're also introducing the host memory (HMEM) based parameter server (PS) along with its SSD-based counterpart. If the sparse model can fit into the host memory of each training node, the optimized HMEM-based PS can provide better model loading and dumping performance with a more effective bandwidth. For more information, refer to [HugeCTR Python Interface](docs/python_interface.md).

+ **Enhancements to the Multi-Nodes TensorFlow Plugin**: The Multi-Nodes TensorFlow Plugin now supports multi-node synchronized training via tf.distribute.MultiWorkerMirroredStrategy. With minimal code changes, you can now easily scale your single GPU training to multi-node multi GPU training. The Multi-Nodes TensorFlow Plugin also supports multi-node synchronized training via Horovod. The inputs for embedding plugins are now data parallel, so the datareader no longer needs to preprocess data for different GPUs based on concrete embedding algorithms. For more information, see our [Sparse Operation Kit Demo](notebooks/sparse_operation_kit_demo.ipynb).
	
+ **NCF Model Support**: We've added support for the NCF model, as well as the GMF and NeuMF variant models. With this enhancement, we're introducing a new element-wise multiplication layer and HitRate evaluation metric. Sample code was added that demonstrates how to preprocess user-item interaction data and train a NCF model with it. New examples have also been added that demonstrate how to train NCF models using MovieLens datasets.

+ **DIN and DIEN Model Support**: All of our layers support the DIN model. The following layers support the DIEN model: FusedReshapeConcat, FusedReshapeConcatGeneral, Gather, GRU, PReLUDice, ReduceMean, Scale, Softmax, and Sub. We also added sample code to demonstrate how to use the Amazon dataset to train the DIN model. See our [DIN sample](samples/din).

+ **Multi-Hot Support for Parquet Datasets**: We've added multi-hot support for parquet datasets, so you can now train models with a paraquet dataset that contains both one hot and multi-hot slots.

+ **Mixed Precision (FP16) Support in More Layers**: The MultiCross layer now supports mixed precision (FP16). All layers now support FP16.

+ **Mixed Precision (FP16) Support in Inference**: We've added FP16 support for the inference pipeline. Therefore, dense layers can now adopt FP16 during inference.

+ **Optimizer State Enhancements for Continuous Training**: You can now store optimizer states that are updated during continuous training as files, such as the Adam optimizer's first moment (m) and second moment (v). By default, the optimizer states are initialized with zeros, but you can specify a set of optimizer state files to recover their previous values. For more information about `dense_opt_states_file` and `sparse_opt_states_file`, refer to [Python Interface](docs/python_interface.md#load_dense_weights-method).

+ **New Library File for GPU Embedding Cache Data**: We’ve moved the header/source code of the GPU embedding cache data structure into a stand-alone folder. It has been compiled into a stand-alone library file. Similar to HugeCTR, your application programs can now be directly linked from this new library file for future use. For more information, refer to our [GPU Embedding Cache ReadMe](gpu_cache/ReadMe.md).

+ **Embedding Plugin Enhancements**: We’ve moved all the embedding plugin files into a stand-alone folder. The embedding plugin can be used as a stand-alone python module, and works with TensorFlow to accelerate the embedding training process.

+ **Adagrad Support**: Adagrad can now be used to optimize your embedding and network. To use it, change the optimizer type in the Optimizer layer and set the corresponding parameters.

## What's New in Version 3.0.1

+ **New DLRM Inference Benchmark**: We've added two detailed Jupyter notebooks to demonstrate how to train, deploy, and benchmark the performance of a deep learning recommendation model (DLRM) with HugeCTR. For more information, refer to our [HugeCTR Inference Notebooks](https://github.com/triton-inference-server/hugectr_backend/tree/v3.0.1-integration/samples/dlrm).

+ **FP16 Optimization**: We've optimized the DotProduct, ELU, and Sigmoid layers based on `__half2` vectorized loads and stores, improving their device memory bandwidth utilization. MultiCross, FmOrder2, ReduceSum, and Multiply are the only layers that still need to be optimized for FP16.

+ **Synthetic Data Generator Enhancements**: We've enhanced our synthetic data generator so that it can generate uniformly distributed datasets, as well as power-law based datasets. You can now specify the `vocabulary_size` and `max_nnz` per categorical feature instead of across all categorial features. For more information, refer to our [user guide](docs/hugectr_user_guide.md#generating-synthetic-data-and-benchmarks).

+ **Reduced Memory Allocation for Trained Model Exportation**: To prevent the "Out of Memory" error message from displaying when exporting a trained model, which may include a very large embedding table, the amount of memory allocated by the related functions has been significantly reduced.

+ **Dropout Layer Enhancement**: The Dropout layer is now compatible with CUDA Graph. The Dropout layer is using cuDNN by default so that it can be used with CUDA Graph.

## What’s New in Version 3.0

+ **Inference Support**: To streamline the recommender system workflow, we’ve implemented a custom HugeCTR backend on the [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server). The HugeCTR backend leverages the embedding cache and parameter server to efficiently manage embeddings of different sizes and models in a hierarchical manner. For more information, refer to [our inference repository](https://github.com/triton-inference-server/hugectr_backend).

+ **New High-Level API**: You can now also construct and train your models using the Python interface with our new high-level API. For more information, refer to [our preview example code](samples/preview) to grasp how this new API works.

+ **[FP16 Support](hugectr_user_guide.md#mixed-precision-training) in More Layers**: All the layers except `MultiCross` support mixed precision mode. We’ve also optimized some of the FP16 layer implementations based on vectorized loads and stores.

+ **[Enhanced TensorFlow Embedding Plugin](notebooks/embedding_plugin.ipynb)**: Our embedding plugin now supports `LocalizedSlotSparseEmbeddingHash` mode. With this enhancement, the DNN model no longer needs to be split into two parts since it now connects with the embedding op through `MirroredStrategy` within the embedding layer.

+ **Extended Embedding Training Cache**: We’ve extended the embedding training cache feature to support `LocalizedSlotSparseEmbeddingHash` and `LocalizedSlotSparseEmbeddingHashOneHot`.

+ **Epoch-Based Training Enhancements**: The `num_epochs` option in the **Solver** clause can now be used with the `Raw` dataset format.

+ **Deprecation of the `eval_batches` Parameter**: The `eval_batches` parameter has been deprecated and replaced with the `max_eval_batches` and `max_eval_samples` parameters. In epoch mode, these parameters control the maximum number of evaluations. An error message will appear when attempting to use the `eval_batches` parameter.

+ **`MultiplyLayer` Renamed**: To clarify what the `MultiplyLayer` does, it was renamed to `WeightMultiplyLayer`.

+ **Optimized Initialization Time**: HugeCTR’s initialization time, which includes the GEMM algorithm search and parameter initialization, was significantly reduced.

+ **Sample Enhancements**: Our samples now rely upon the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) instead of the Kaggle Display Advertising Challenge dataset. Our preprocessing scripts (Perl, Pandas, and NVTabular) have also been unified and simplified.

+ **Configurable DataReader Worker**: You can now specify the number of data reader workers, which run in parallel, with the `num_workers` parameter. Its default value is 12. However, if you are using the Parquet data reader, you can't configure the `num_workers` parameter since it always corresponds to the number of active GPUs.

## What's New in Version 2.3

+ **New Python Interface**: To enhance the interoperability with [NVTabular](https://github.com/NVIDIA/NVTabular) and other Python-based libraries, we're introducing a new Python interface for HugeCTR.

+ **HugeCTR Embedding with Tensorflow**: To help users easily integrate HugeCTR’s optimized embedding into their Tensorflow workflow, we now offer the HugeCTR embedding layer as a Tensorflow plugin. To better understand how to intall, use, and verify it, see our [Jupyter notebook tutorial](../notebooks/embedding_plugin.ipynb). It also demonstrates how you can create a new Keras layer, `EmbeddingLayer`, based on the [`hugectr.py`](../tools/embedding_plugin/python) helper code that we provide.

+ **Embedding Training Cache**: To enable a model with large embedding tables that exceeds the single GPU's memory limit, we've added a new embedding training cache feature, giving you the ability to load a subset of an embedding table into the GPU in a coarse grained, on-demand manner during the training stage.

+ **TF32 Support**: We've added TensorFloat-32 (TF32), a new math mode and third-generation of Tensor Cores, support on Ampere. TF32 uses the same 10-bit mantissa as FP16 to ensure accuracy while providing the same range as FP32 by using an 8-bit exponent. Since TF32 is an internal data type that accelerates FP32 GEMM computations with tensor cores, you can simply turn it on with a newly added configuration option. For more information, refer to [Solver](docs/hugectr_user_guide.md#solver).

+ **Enhanced AUC Implementation**: To enhance the performance of our AUC computation on multi-node environments, we've redesigned our AUC implementation to improve how the computational load gets distributed across nodes.

+ **Epoch-Based Training**: In addition to the `max_iter` parameter, you can now set the `num_epochs` parameter in the **Solver** clause within the configuration file. This mode can only currently be used with `Norm` dataset formats and their corresponding file lists. All dataset formats will be supported in the future.

+ **New Multi-Node Training Tutorial**: To better support multi-node training use cases, we've added a new [a step-by-step tutorial](../tutorial/multinode-training).

+ **Power Law Distribution Support with Data Generator**: Because of the increased need for generating a random dataset whose categorical features follows the power-law distribution, we've revised our data generation tool to support this use case. For additional information, refer to the `--long-tail` description [here](../docs/hugectr_user_guide.md#Generating Synthetic Data and Benchmarks).

+ **Multi-GPU Preprocessing Script for Criteo Samples**: Multiple GPUs can now be used when preparing the dataset for our [samples](../samples). For more information, see how [preprocess_nvt.py](../tools/criteo_script/preprocess_nvt.py) is used to preprocess the Criteo dataset for DCN, DeepFM, and W&D samples.

## Known Issues

+ HugeCTR uses NCCL to share data between ranks, and NCCL may require shared system memory for IPC and pinned (page-locked) system memory resources. When using NCCL inside a container, it is recommended that you increase these resources by issuing: `-shm-size=1g -ulimit memlock=-1`
See also [NCCL's known issue](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data). And the [GitHub issue](https://github.com/NVIDIA-Merlin/HugeCTR/issues/243).

+ Softmax layer currently does not support fp16 mode.

+ KafkaProducers startup will succeed, even if the target Kafka broker is unresponsive. In order to avoid data-loss in conjunction with streaming model updates from Kafka, you have to make sure that a sufficient number of Kafka brokers is up, working properly and reachable from the node where you run HugeCTR.

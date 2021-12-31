# Release Notes
## What's New in Version 3.3.1
+ **Hierarchical Parameter Server Enhancements**:
    + **Online deployment of new models and recycling of old models**: In this release, HugeCTR Backend is fully compatible with the [model control protocol](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md) of Triton. Adding the configuration of a new model to the [HPS configuration file](https://github.com/triton-inference-server/hugectr_backend#independent-parameter-server-configuration). The HugeCTR Backend has supported online deployment of new models by the Load API of Triton. The old models can also be recycled online by the [Unload API](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md#unload).
    + **Simplified database backend**: Multi-nodes, single-node and all other kinds of volatile database backends can now be configured using the same configuration object.
    + **Multi-threaded optimization of Redis code**: ~2.3x speedup up over HugeCTR v3.3.
    + **Fix to some issues**: Build HPS test environment and implement unit test of each component; Access violation issue of online Kafka updates; Parquet data reader incorrectly parses the index of categorical features in the case of multiple embedded tables; HPS Redis Backend overflow handling not invoked upon single insertions.
+ **New group of fused fully connected layers**: We support adding a group of fused fully connected layers when constructing the model graph. A concise Python interface is provided for users to adjust the number of layers, as well as to specify the output dimensions in each layer, which makes it easy to leverage the highly-optimized fused fully connected layer in HugeCTR. For more information, please refer to [GroupDenseLayer](docs/python_interface.md#groupdenselayer)
+ **Fix to some issues**: 
   + Warnning is added for the case users forget to import mpi before launching multi-process job
   + Removing massive log when runing with embedding training cache
   + Removing lagacy conda related informations from documents

## What's New in Version 3.3

+ **Hierarchical Parameter Server**: 
    + **Support Incremental Models Updating From Online Training**: HPS now supports iterative model updating via Kafka message queues. It is now possible to connect HugeCTR with Apache Kafka deployments to update the model in-place in real-time. This feature is supported in both phases, training and inference. Please refer to the [Demo Notebok](https://github.com/triton-inference-server/hugectr_backend/tree/main/samples/hierarchical_deployment/hps_e2e_demo).
    + **Support Embedding keys Eviction Mechanism**: In-memory databases such as Redis or CPU memory backed storage are used now as the feature memory management. Hence, when performing iterative updating, they will automatically evict infrequently used embeddings as training progresses.
    + **Support Embedding Cache Asynchronous Refresh Mechanism**: We have supported the asynchronous refreshing of incremental embedding keys into the embedding cache. Refresh operation will be triggered when completing the model version iteration or incremental parameters output from online training. The Distributed Database and Persistent Database will be updated by the distributed event streaming platform(Kafka). And then the GPU embedding cache will refresh the values of the existing embedding keys and replace them with the latest incremental embedding vectors. Please refer to the [HPS README](https://github.com/triton-inference-server/hugectr_backend#hugectr-hierarchical-parameter-server).
    + **Other Improvements**: Backend implementations for databases are now fully configurable. JSON interface parser can cope better with inaccurate parameterization. Less and if (hopefully) more meaningful jabber! Based on your requests, we revised the log levels for throughout the entire database backend API of the parameter server. Selected configuration options are now printed wholesomely and uniformly to the log. Errors provide more verbose information on the matter at hand. Improved performance of Redis cluster backend. Improved performance of CPU memory database backend.

+ **SOK TF 1.15 Support**: In this version, SOK can be used along with TensorFlow 1.15. See [README](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/get_started/get_started.html#tensorflow-1-15). Dedicated CUDA stream is used for SOK’s Ops, and kernel interleaving might be eliminated. Users can now install SOK via pip install SparseOperationKit, which no longer requires root access to compile SOK and no need to copy python scripts. There was a hanging issue in `tf.distribute.MirroredStrategy` when TensorFlow version greater than 2.4. In this version, this issue in TensorFlow 2.5+ is fixed.

+ **MLPerf v1.1 integration**：
    + **Hybrid-embedding indices pre-computing**：The indices needed for hybrid embedding are pre-computed ahead of time and are overlapped with previous iterations.
    + **Cached evaluation indices:**：The hybrid-embedding indices for eval are cached when applicable, hence eliminating the re-computing of the indices at every eval iteration.
    + **MLP weight/data gradients calculation overlap:**：The weight gradients of MLP are calculated asynchronously with respect to the data gradients, enabling overlap between these two computations.
    + **Better compute-communication overlap:**：Better overlap between compute and communication has been enabled to improve training throughput.
    + **Fused weight conversion:**：The FP32-to-FP16 conversion of the weights are now fused into the SGD optimizer, saving trips to memory.
    + **GraphScheduler:**：GrapScheduler was added to control the timing of cudaGraph launching. With GraphScheduler, the gap between adjacent cudaGraphs is eliminated.

+ **Multi-node training support on the cluster without RDMA**：We support multi-node training without RDMA now. You can specify allreduce algorithm as `AllReduceAlgo.NCCL` and it can support non-RDMA hardware. For more information, please refer to `all_reduce_algo` in [CreateSolver API](docs/python_interface.md#createsolver-method).

+ **SOK support device setting with tf.config**：`tf.config.set_visible_device` can be used to set the visible GPUs for each process. Meanwhile, `CUDA_VISIBLE_DEVICES` can also be used to achieve the same purpose. When `tf.distribute.Strategy` is used, device argument must not be set.

+ **User defined name is supported in model dumping**: We support specifying the model name with the training API `CreateSolver`, which will be dumped to the JSON configuration file with the API `Model.graph_to_json`. This feature will facilitate the Triton deployment of saved HugeCTR models, and help to distinguish between models when Kafka sends parameters from the training side to the inference side.

+ **Fine-grained control of the embedding layers**: We support the fine-grained control of the embedding layers. Users can freeze or unfreeze the weights of a specific embedding layer with the APIs `Model.freeze_embedding` and `Model.unfreeze_embedding`. Besides, the weights of multiple embedding layers can be loaded independently, which enables the use case of loading pre-trained embeddings for a particular layer. For more information, please refer to [Model API](docs/python_interface.md#model) and Section 3.4 of [HugeCTR Criteo Notebook](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/notebooks/hugectr_criteo.ipynb).



## What's New in Version 3.2.1

+ **GPU Embedding Cache Optimization**: The performance of the GPU embedding cache for the standalone module has been optimized. With this enhancement, the performance of small to medium batch sizes has improved significantly. We're not introducing any changes to the interface for the GPU embedding cache, so don't worry about making changes to any existing code that uses this standalone module. For more information, refer to the [GPU Cache ReadMe](./gpu_cache/ReadMe.md).

+ **Model Oversubscription Enhancements**: We're introducing a new host memory cache (HMEM-Cache) component for the model oversubscription feature. When configured properly, incremental training can be efficiently performed on models with large embedding tables that exceed the host memory. For more information, refer to [Host Memory Cache in MOS](./docs/intro_hmem_cache.md). Additionally, we've enhanced the Python interface for model oversubscription by replacing the `use_host_memory_ps` parameter with a `ps_types` parameter and adding a `sparse_models` parameter. For more information about these changes, refer to [HugeCTR Python Interface](./docs/python_interface.md#createmos-method).

+ **Debugging Enhancements**: We're introducing new debugging features such as multi-level logging, as well as kernel debugging functions. We're also making our error messages more informative so that users know exactly how to resolve issues related to their training and inference code. For more information, refer to the comments in the header files, which are available at HugeCTR/include/base/debug.

+ **Enhancements to the Embedding Key Insertion Mechanism for the Embedding Cache**: Missing embedding keys can now be asynchronously inserted into the embedding cache. To enable automatically, set the hit rate threshold within the configuration file. When the actual hit rate of the embedding cache is higher than the hit rate threshold that the user set or vice versa, the embedding cache will insert the missing embedding key asynchronously.

+ **Parameter Server Enhancements**: We're introducing a new "in memory" database that utilizes the local CPU memory for storing and recalling embeddings and uses multi-threading to accelerate lookup and storage. You can now also use the combined CPU-accessible memory of your Redis cluster to store embeddings. We improved the performance for the "persistent" storage and retrieving embeddings from RocksDB using structured column families, as well as added support for creating hierarchical storage such as Redis as distributed cache. You don't have to worry about updating your Parameter Server configurations to take advantage of these enhancements.

+ **Slice Layer Internalization Enhancements**: The Slice layer for the branch toplogy can now be abstracted away in the Python interface. A model graph analysis will be conducted to resolve the tensor dependency and the Slice layer will be internally inserted if the same tensor is consumed more than once to form the branch topology. For more information about how to construct a model graph using branches without the Slice layer, refer to [Getting Started](README.md#getting-started) and [Slice Layer](./docs/hugectr_layer_book.md#slice-layer).

## What's New in Version 3.2

+ **New HugeCTR to ONNX Converter**: We’re introducing a new HugeCTR to ONNX converter in the form of a Python package. All graph configuration files are required and model weights must be formatted as inputs. You can specify where you want to save the converted ONNX model. You can also convert sparse embedding models. For more information, refer to [HugeCTR to ONNX Converter](./onnx_converter) and [HugeCTR2ONNX Demo Notebook](notebooks/hugectr2onnx_demo.ipynb).

+ **New Hierarchical Storage Mechanicsm on the Parameter Server (POC)**: We’ve implemented a hierarchical storage mechanism between local SSDs and CPU memory. As a result, embedding tables no longer have to be stored in the local CPU memory. The distributed Redis cluster is being implemented as a CPU cache to store larger embedding tables and interact with the GPU embedding cache directly. The local RocksDB serves as a query engine to back up the complete embedding table on the local SSDs and assist the Redis cluster with looking up missing embedding keys. For more information about how this works, refer to our [HugeCTR Backend documentation](https://github.com/triton-inference-server/hugectr_backend/blob/main/docs/architecture.md#distributed-deployment-with-hierarchical-hugectr-parameter-server)

+ **Parquet Format Support within the Data Generator**: The HugeCTR data generator now supports the parquet format, which can be configured easily using the Python API. For more information, refer to [Data Generator API](docs/python_interface.md#data-generator-api).

+ **Python Interface Support for the Data Generator**: The data generator has been enabled within the HugeCTR Python interface. The parameters associated with the data generator have been encapsulated into the `DataGeneratorParams` struct, which is required to initialize the `DataGenerator` instance. You can use the data generator's Python APIs to easily generate the Norm, Parquet, or Raw dataset formats with the desired distribution of sparse keys. For more information, refer to [Data Generator API](docs/python_interface.md#data-generator-api) and [Data Generator Samples](tools/data_generator).

+ **Improvements to the Formula of the Power Law Simulator within the Data Generator**: We've modified the formula of the power law simulator within the data generator so that a positive alpha value is always produced, which will be needed for most use cases. The alpha values for `Long`, `Medium`, and `Short` within the power law distribution are 0.9, 1.1, and 1.3 respectively. For more information, refer to [Data Generator API](docs/python_interface.md#data-generator-api).

+ **Support for Arbitrary Input and Output Tensors in the Concat and Slice Layers**: The Concat and Slice layers now support any number of input and output tensors. Previously, these layers were limited to a maximum of four tensors.

+ **New Continuous Training Notebook**: We’ve added a new notebook to demonstrate how to perform continuous training using the embedding training cache (also referred to as Embedding Training Cache) feature. For more information, refer to [HugeCTR Continuous Training](notebooks/continuous_training.ipynb).

+ **New HugeCTR Contributor Guide**: We've added a new [HugeCTR Contributor Guide](docs/hugectr_contributor_guide.md) that explains how to contribute to HugeCTR, which may involve reporting and fixing a bug, introducing a new feature, or implementing a new or pending feature.

+ **Enhancements to Sparse Operation Kits (SOK)**: SOK now supports TensorFlow 2.5 and 2.6. We also added support for identity hashing, dynamic input, and Horovod within SOK. Lastly, we added a new [SOK docs set](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/v1.0.0/index.html) to help you get started with SOK.

## What's New in Version 3.1

+ **MLPerf v1.0 Integration**: We've integrated MLPerf optimizations for DLRM training and enabled them as configurable options in Python interface. Specifically, we have incorporated AsyncRaw data reader, HybridEmbedding, FusedReluBiasFullyConnectedLayer, overlapped pipeline, holistic CUDA Graph and so on. The performance of 14-node DGX-A100 DLRM training with Python APIs is comparable to CLI usage. For more information, refer to [HugeCTR Python Interface](docs/python_interface.md) and [DLRM Sample](samples/dlrm).

+ **Enhancements to the Python Interface**: We’ve enhanced the Python interface for HugeCTR so that you no longer have to manually create a JSON configuration file. Our Python APIs can now be used to create the computation graph. They can also be used to dump the model graph as a JSON object and save the model weights as binary files so that continuous training and inference can take place. We've added an Inference API that takes Norm or Parquet datasets as input to facilitate the inference process. For more information, refer to [HugeCTR Python Interface](docs/python_interface.md) and [HugeCTR Criteo Notebook](notebooks/hugectr_criteo.ipynb).

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

+ **Synthetic Data Generator Enhancement**: We've enhanced our synthetic data generator so that it can generate uniformly distributed datasets, as well as power-law based datasets. You can now specify the `vocabulary_size` and `max_nnz` per categorical feature instead of across all categorial features. For more information, refer to our [user guide](docs/hugectr_user_guide.md#generating-synthetic-data-and-benchmarks).

+ **Reduced Memory Allocation for Trained Model Exportation**: To prevent the "Out of Memory" error message from displaying when exporting a trained model, which may include a very large embedding table, the amount of memory allocated by the related functions has been significantly reduced.

+ **Dropout Layer Enhancement**: The Dropout layer is now compatible with CUDA Graph. The Dropout layer is using cuDNN by default so that it can be used with CUDA Graph.

## What’s New in Version 3.0

+ **Inference Support**: To streamline the recommender system workflow, we’ve implemented a custom HugeCTR backend on the [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server). The HugeCTR backend leverages the embedding cache and parameter server to efficiently manage embeddings of different sizes and models in a hierarchical manner. For more information, refer to [our inference repository](https://github.com/triton-inference-server/hugectr_backend).

+ **New High-Level API**: You can now also construct and train your models using the Python interface with our new high-level API. For more information, refer to [our preview example code](samples/preview) to grasp how this new API works.

+ **[FP16 Support](hugectr_user_guide.md#mixed-precision-training) in More Layers**: All the layers except `MultiCross` support mixed precision mode. We’ve also optimized some of the FP16 layer implementations based on vectorized loads and stores.

+ **[Enhanced TensorFlow Embedding Plugin](notebooks/embedding_plugin.ipynb)**: Our embedding plugin now supports `LocalizedSlotSparseEmbeddingHash` mode. With this enhancement, the DNN model no longer needs to be split into two parts since it now connects with the embedding op through `MirroredStrategy` within the embedding layer.

+ **Extended Embedding Training Cache**: We’ve extended the embedding training cache feature to support `LocalizedSlotSparseEmbeddingHash` and `LocalizedSlotSparseEmbeddingHashOneHot`.

+ **Epoch-Based Training Enhancement**: The `num_epochs` option in the **Solver** clause can now be used with the `Raw` dataset format.

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
+ Since the automatic plan file generator isn't able to handle systems that contain one GPU, you must manually create a JSON plan file with the following parameters and rename it using the name listed in the HugeCTR configuration file: `{"type": "all2all", "num_gpus": 1, "main_gpu": 0, "num_steps": 1, "num_chunks": 1, "plan": [[0, 0]], and "chunks": [1]}`.

+ If using a system that contains two GPUs with two NVLink connections, the auto plan file generator will print the following warning message: `RuntimeWarning: divide by zero encountered in true_divide`. This is an erroneous warning message and should be ignored.

+ The current plan file generator doesn't support a system where the NVSwitch or a full peer-to-peer connection between all nodes is unavailable.

+ Users need to set an `export CUDA_DEVICE_ORDER=PCI_BUS_ID` environment variable to ensure that the CUDA runtime and driver have a consistent GPU numbering.

+ `LocalizedSlotSparseEmbeddingOneHot` only supports a single-node machine where all the GPUs are fully connected such as NVSwitch.

+ HugeCTR version 3.0 crashes when running the DLRM sample on DGX2 due to a CUDA Graph issue. To run the sample on DGX2, disable the CUDA Graph by setting the `cuda_graph` parameter to false even if it degrades the performance a bit. This issue doesn't exist when using the DGX A100.

+ The HugeCTR embedding TensorFlow plugin only works with single-node machines.

+ The HugeCTR embedding TensorFlow plugin assumes that the input keys are in `int64` and its output is in `float`.

+ If the number of samples in a dataset is not divisible by the batch size when in epoch mode and using the `num_epochs` instead of `max_iter`, a few remaining samples are truncated. If the training dataset is large enough, its impact can be negligible. If you want to minimize the wasted batches, try adjusting the number of data reader workers. For example, using a file list source, set the `num_workers` parameter to an advisor based on the number of data files in the file list.

+ The MultiCross layer doesn't support mixed precision mode yet.

+ Asynchronously inserts missed embeddings into the in-memory database backend.
Currently, we only fix misses in the database backend through the insertion logic via Kafka. That means if the model is not updated, we also do not improve caching. We need to establish an asynchronous process that also inserts lookup-misses into the database backend.

+ Kafka producer's asynchronous connection built-up may lead to message loss.
Rdkafka client checks connections are performed asynchronously in the background. During this period, the application thinks it is connected and keeps producing data. These messages will never reach Kafka and will be lost.

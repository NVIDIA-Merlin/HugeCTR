# Release Notes

## What's New in Version 3.1
MLPerf v1.0 DLRM benchmark 
+ **Python Interface Enhancements (drop json config file)**: We’ve enhanced the Python interface for HugeCTR, which supports constructing the computation graph with Python APIs and saves users from the effort of manually writing the JSON configuration file. We also provide APIs to dump the model graph to JSON and to save the model weights to binary files, which can be used for both continuous training and inference. Besides, an inference API that takes Norm or Parquet dataset as input is provided to facilitate the inference work. Please refer to [HugeCTR Python Interface](docs/python_interface.md) to get familiar with the APIs and [HugeCTR Criteo Notebook](notebooks/hugectr_criteo.ipynb) to see the usage.

+ **Unified Embedding**: We’re introducing a new interface for embedding and datareader to simplify the use of embedding and datareader. For datareader, we now provide `nnz_per_slot` and `is_fixed_length` to help you specify the number of keys in each slot. For embedding, you can now directly configure how much memory usage you need by specifying `workspace_size_per_gpu_in_mb` instead of the origin of `max_vocabulary_size_per_gpu`. Now we use `mean/sum` in combinators instead of numbers 0/1, which is more convenient to use. You can learn how to use the new interface in [HugeCTR Python Interface](docs/python_interface.md).

+ **Multi-nodes Embedding training Cache (MOS)**: We’ve enabled multi-node support for the embedding training cache. This will allow you to train a model with a terabyte size embedding table using a single or several nodes even though the whole embedding table can not fit into the GPU memory. Besides, we introduced the host memory (HMEM)-based parameter server (PS) along with its SSD-based counterpart, and the optimized HMEM-based PS can give better performance (>5x higher effective bandwidth) of model loading and dumping if the sparse model can fit into the host memory of all training nodes. Please refer to the [HugeCTR Python Interface](docs/python_interface.md) to learn more about the interface and configuration details.

+ **Multi-nodes TF Plugin**: It supports Multi-node synchronized training via tf.distribute.MultiWorkerMirroredStrategy. With little code changing, you can scale up your training process from single GPU training to multi-node multi GPU training. We will provide detailed examples and references in the near future.
Besides tf.distribute.MultiWorkerMirroredStrategy, it supports multi-node synchronized training via Horovod. We will provide usage examples in the near future. From now on, the inputs to embedding plugins are data-parallel rather than GPU-specific datas, which means the datareader no longer needs to preprocess datas for different GPUs based on concrete embedding algorithms.
	
+ **NCF model support**: We have added support for the NCF model and two variants, the GMF and NeuMF models. This support includes a new element-wise multiplication layer and a new HitRate evaluation metric. Sample code was added that demonstrates how to preprocess user-item interaction data and use it to train an NCF model, with examples given using MovieLens datasets.

+ **Supporting DIN model**: 

+ **Supporting multi-hot in parquet data reader**: We've added multi-hot support of parquet dataset file. You can train models with a dataset containing both one hot and multi-hot slots. Currently, only iteration-based training mode is supported with parquet dataset format.

+ **Supporting Mixed-precision in more layers**: The MultiCross layer now supports mixed-precision (FP16). As a result, all layers now support mixed-precision.

+ **Supporting Optimizer States Save/Load for continued training**: You can choose to store optimizer states updated during the training to files like you save trained weights of your model. For instance, Adam optimizer has the first moment (m) and the second moment (v), which keep being updated across training iterations. By default, they are initialized with zeros, but you can specify a set of optimizer state files to recover their previous values. For more details, find `dense_opt_states_file` and `sparse_opt_states_file` in  [Python Interface](docs/python_interface.md#load_dense_weights-method).

+ **Supporting FP16 in inference**: We have supported FP16 for the inference pipeline which means that half precision forward propagation can be adopted by the dense layers during inference.

+ **Embedding cache release in submodule**: We’ve separated the header/source code/document of GPU embedding cache data structure into a stand-alone folder. Now it will be compiled into a stand-alone library file, so your application programs can be directly linked against it just as HugeCTR does.   

+ **Embedding plugin release in submodule**: We’ve separated all the files related to embedding plugin into a stand-along folder. It can be used as a stand-alone python module, and works with TensorFlow to accelerate the embedding training process.

+ **Supporting Adagrad**: HugeCTR now supports Adagrad to optimize your embedding and network now. You can use it just by changing the optimizer type in `Optimizer` and set corresponding parameters.


## What's New in Version 3.0.1

+ **DLRM Inference Benchmark**: We've added two detailed Jupyter notebooks to illustrate how to train and deploy a DLRM model with HugeCTR whilst benchmarking its performance. The inference notebook demonstrates how to create Triton and HugeCTR backend configs, prepare the inference data, and deploy a trained model by another notebook on Triton Inference Server. It also shows the way of benchmarking its performance (throughput and latency), based on Triton Performance Analyzer. For more details, check out our [HugeCTR inference repository](https://github.com/triton-inference-server/hugectr_backend/tree/v3.0.1-integration/samples/dlrm).
+ **FP16 Speicific Optimization in More Dense Layers**: We've optimized DotProduct, ELU, and Sigmoid layers based on `__half2` vectorized loads and stores, so that they better utilize device memory bandwidth. Now most layers have been optimized in such a way except MultiCross, FmOrder2, ReduceSum, and Multiply layers.
+ **More Finely Tunable Synthetic Data Generator**: Our new data generator can generate uniformly distributed datasets in addition to power law based datasets. Instead of specifying `vocabulary_size` in total and `max_nnz`, you can specify such information per categorical feature. See [our user guide](docs/hugectr_user_guide.md#generating-synthetic-data-and-benchmarks) to learn its changed usage.
+ **Decreased Memory Demands of Trained Model Exportation**: To prevent the out of memory error from happening in saving a trained model including a very large embedding table, the actual amount of memory allocated by the related functions was effectively reduced.
+ **CUDA Graph Compatible Dropout Layer**: HugeCTR Dropout Layer uses cuDNN by default, so that it can be used together with CUDA Graph. In the previous version, if Dropout was used, CUDA Graph was implicitly turned off.

## What’s New in Version 3.0

+ **Inference Support**: To streamline the recommender system workflow, we’ve implemented a custom HugeCTR backend on the [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server). The HugeCTR backend leverages the embedding cache and parameter server to efficiently manage embeddings of different sizes and models in a hierarchical manner. For additional information, see [our inference repository](https://github.com/triton-inference-server/hugectr_backend).

+ **New High-Level API**: You can now also construct and train your models using the Python interface with our new high-level API. See [our preview example code](samples/preview) to grasp how it works.

+ **[FP16 Support](hugectr_user_guide.md#mixed-precision-training) in More Layers**: All the layers except `MultiCross` support mixed precision mode. We’ve also optimized some of the FP16 layer implementations based on vectorized loads and stores.

+ **[Enhanced TensorFlow Embedding Plugin](notebooks/embedding_plugin.ipynb)**: Our embedding plugin now supports `LocalizedSlotSparseEmbeddingHash` mode. With this enhancement, the DNN model no longer needs to be split into two parts since it now connects with the embedding op through `MirroredStrategy` within the embedding layer.

+ **Extended Model Oversubscription**: We’ve extended the model oversubscription feature to support `LocalizedSlotSparseEmbeddingHash` and `LocalizedSlotSparseEmbeddingHashOneHot`.

+ **Epoch-Based Training Enhancement**: The `num_epochs` option in the **Solver** clause can now be used with the `Raw` dataset format.

+ **Deprecation of the `eval_batches` Parameter**: The `eval_batches` parameter has been deprecated and replaced with the `max_eval_batches` and `max_eval_samples` parameters. In epoch mode, these parameters control the maximum number of evaluations. An error message will appear when attempting to use the `eval_batches` parameter.

+ **`MultiplyLayer` Renamed**: To clarify what the `MultiplyLayer` does, it was renamed to `WeightMultiplyLayer`.

+ **Optimized Initialization Time**: HugeCTR’s initialization time, which includes the GEMM algorithm search and parameter initialization, was significantly reduced.

+ **Sample Enhancements**: Our samples now rely upon the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) instead of the Kaggle Display Advertising Challenge dataset. Our preprocessing scripts (Perl, Pandas, and NVTabular) have also been unified and simplified.

+ **Configurable DataReader Worker**: You can now specify the number of data reader workers, which run in parallel, with the `num_workers` parameter. Its default value is 12. However, if you are using the Parquet data reader, you can't configure the `num_workers` parameter since it always corresponds to the number of active GPUs.

## What's New in Version 2.3

+ **New Python Interface**: To enhance the interoperability with [NVTabular](https://github.com/NVIDIA/NVTabular) and other Python-based libraries, we're introducing a new Python interface for HugeCTR.

+ **HugeCTR Embedding with Tensorflow**: To help users easily integrate HugeCTR’s optimized embedding into their Tensorflow workflow, we now offer the HugeCTR embedding layer as a Tensorflow plugin. To better understand how to intall, use, and verify it, see our [Jupyter notebook tutorial](../notebooks/embedding_plugin.ipynb). It also demonstrates how you can create a new Keras layer, `EmbeddingLayer`, based on the [`hugectr.py`](../tools/embedding_plugin/python) helper code that we provide.

+ **Model Oversubscription**: To enable a model with large embedding tables that exceeds the single GPU's memory limit, we've added a new model oversubscription feature, giving you the ability to load a subset of an embedding table into the GPU in a coarse grained, on-demand manner during the training stage.

+ **TF32 Support**: We've added TensorFloat-32 (TF32), a new math mode and third-generation of Tensor Cores, support on Ampere. TF32 uses the same 10-bit mantissa as FP16 to ensure accuracy while providing the same range as FP32 by using an 8-bit exponent. Since TF32 is an internal data type that accelerates FP32 GEMM computations with tensor cores, you can simply turn it on with a newly added configuration option. For additional information, see [Solver](docs/hugectr_user_guide.md#solver).

+ **Enhanced AUC Implementation**: To enhance the performance of our AUC computation on multi-node environments, we've redesigned our AUC implementation to improve how the computational load gets distributed across nodes.

+ **Epoch-Based Training**: In addition to the `max_iter` parameter, you can now set the `num_epochs` parameter in the **Solver** clause within the configuration file. This mode can only currently be used with `Norm` dataset formats and their corresponding file lists. All dataset formats will be supported in the future.

+ **New Multi-Node Training Tutorial**: To better support multi-node training use cases, we've added a new [a step-by-step tutorial](../tutorial/multinode-training).

+ **Power Law Distribution Support with Data Generator**: Because of the increased need for generating a random dataset whose categorical features follows the power-law distribution, we've revised our data generation tool to support this use case. For additional information, refer to the `--long-tail` description [here](../docs/hugectr_user_guide.md#Generating Synthetic Data and Benchmarks).

+ **Multi-GPU Preprocessing Script for Criteo Samples**: Multiple GPUs can now be used when preparing the dataset for our [samples](../samples). For additional information, see how [preprocess_nvt.py](../tools/criteo_script/preprocess_nvt.py) is used to preprocess the Criteo dataset for DCN, DeepFM, and W&D samples.

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

+ MultiCross layer doesn't support mixed precision mode yet.

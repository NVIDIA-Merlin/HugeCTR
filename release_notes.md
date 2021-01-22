# Release Notes

## What's New in Version 2.3
We’ve implemented the following enhancements to improve usability and performance:
+ **Python Interface**: To enhance the interoperability with [NVTabular](https://github.com/NVIDIA/NVTabular) and other Python-based libraries, we're introducing a new Python interface for HugeCTR.

+ **HugeCTR Embedding with Tensorflow**: To help users easily integrate HugeCTR’s optimized embedding into their Tensorflow workflow, we now offer the HugeCTR embedding layer as a Tensorflow plugin. To better understand how to intall, use, and verify it, see our [Jupyter notebook tutorial](../notebooks/embedding_plugin.ipynb). It also demonstrates how you can create a new Keras layer `EmbeddingLayer` based on the [`hugectr.py`](../tools/embedding_plugin/python) helper code that we provide.

+ **Model Oversubscription**: To enable a model with large embedding tables that exceeds the single GPU's memory limit, we added a new model oversubscription feature, giving you the ability to load a subset of an embedding table into the GPU in a coarse grained, on-demand manner during the training stage.

+ **TF32 Support**: The third-generation Tensor Cores on Ampere support, TensorFloat-32 (TF32), a novel math mode. TF32 uses the same 10-bit mantissa as FP16 to ensure accuracy while providing the same range as FP32 by using an 8-bit exponent. Since TF32 is an internal data type that accelerates FP32 GEMM computations with tensor cores, a user can simply turn it on with a newly added configuration option. For additional information, see [Solver](docs/hugectr_user_guide.md#solver).

+ **Enhanced AUC Implementation**: To enhance the performance of our AUC computation on multi-node environments, we redesigned our AUC implementation to improve how the computational load gets distributed across nodes.

+ **Epoch-Based Training**: In addition to `max_iter`, a HugeCTR user can set `num_epochs` in the **Solver** clause of their configuration file. This mode can only currently be used with `Norm` dataset formats and their corresponding file lists. All dataset formats will be supported in the future.

+ **Multi-Node Training Tutorial**: To better support multi-node training use cases, we added a new [a step-by-step tutorial](../tutorial/multinode-training).

+ **Power Law Distribution Support with Data Generator**: Because of the increased need for generating a random dataset whose categorical features follows the power-law distribution, we revised our data generation tool to support this use case. For additional information, refer to the `--long-tail` description [here](../docs/hugectr_user_guide.md#Generating Synthetic Data and Benchmarks).

+ **Multi-GPU Preprocessing Script for Criteo Samples**: Multiple GPUs can now be used when preparing the dataset for our [samples](../samples). For additional information, see how [preprocess_nvt.py](../tools/criteo_script/preprocess_nvt.py) is used to preprocess the Criteo dataset for DCN, DeepFM, and W&D samples.

## Known Issues
* Since the automatic plan file generator is not able to handle systems that contain one GPU, a user must manually create a JSON plan file with the following parameters and rename using the name listed in the HugeCTR configuration file: ` {"type": "all2all", "num_gpus": 1, "main_gpu": 0, "num_steps": 1, "num_chunks": 1, "plan": [[0, 0]], "chunks": [1]} `.
* If using a system that contains two GPUs with two NVLink connections, the auto plan file generator will print the following warning message: `RuntimeWarning: divide by zero encountered in true_divide`. This is an erroneous warning message and should be ignored.
* The current plan file generator doesn't support a system where the NVSwitch or a full peer-to-peer connection between all nodes is unavailable.
* Users need to set an `export CUDA_DEVICE_ORDER=PCI_BUS_ID` environment variable to ensure that the CUDA runtime and driver have a consistent GPU numbering.
* `LocalizedSlotSparseEmbeddingOneHot` only supports a single-node machine where all the GPUs are fully connected such as NVSwitch.
* HugeCTR version 2.2.1 crashes when running our DLRM sample on DGX2 due to a CUDA Graph issue. To run the sample on DGX2, disable the use of CUDA Graph with `"cuda_graph": false` even if it degrades the performance a bit. We are working on fixing this issue. This issue doesn't exist when using the DGX A100.
* The model oversubscription feature is only available in Python. Currently a user can only use this feature with the `DistributedSlotSparseEmbeddingHash` embedding and the `Norm` dataset format on single GPUs. This feature will eventually support all embedding types and dataset formats.
* The HugeCTR embedding TensorFlow plugin only works with single-node machines.
* The HugeCTR embedding TensorFlow plugin assumes that the input keys are in `int64` and its output is in `float`.
* When using our embedding plugin, please note that the `fprop_v3` function, which is available in `tools/embedding_plugin/python/hugectr.py`, only works with `DistributedSlotSparseEmbeddingHash`.
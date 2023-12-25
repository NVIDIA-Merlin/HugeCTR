# SparseOperationKit Release Notes #
The release notes for SparseOperationKit.

## What's new in Version 2.0.0 ##
+ The official release of SOK:
    * Remove the legacy code,`DistributedEmbedding` and `All2AllDenseEmbedding` will be deprecated.
    * `from sparse_operation_kit import experiment as sok` is replaced by `import sparse_operation_kit as sok`.
    * `sok.DynamicVariable` supports Merlin-HKV as its backend.
    * The parallel dump and load are added.

## What's new in Version 1.1.4 ##
+ Add `sok.experiment` module to integrate hugectr 3G embedding:
    * Add `sok.experiment.lookup_sparse`, which support distributed and fused embedding lookup.
    * Add `sok.experiment.DynamicVariable`, whose size can grow dynamically when doing lookup.
    * See API Docs -> Experiment to get other function of `sok.experiment`

## What's new in Version 1.1.3 ##
+ Update pip install instruction and fix some bugs.

## What's new in Version 1.1.2 ##
+ Add TensorFlow Functional API support

## What's new in Version 1.1.1 ##
+ Add Auto-Mixed-Precision training support
+ Add uint32 key dtype support
+ Add TensorFlow initializers support
+ Add DLRM benchmark results

## What's new in Version 1.1.0 ##
+ Supports TensorFlow 1.15.
+ Supports configuring visible devices via `tf.config.set_visible_devices`.
+ Added a dedicated CUDA stream for SOK's Ops.
+ Supports pip installation.
+ Fixed hanging issue in `tf.distribute.MirroredStrategy` when TensorFlow version greater than 2.4.

## What's new in Version 1.0.1 ##
+ Supports Horovod as the synchronized training communication tool.
+ Supports dynamic input in All2AllDenseEmbedding, which means `unique->lookup->gather` pattern can be used.
+ Supports IdentityHashtable, which means no hash-mapping during inserting new keys.
+ Added TF Distributed Embedding totally with TF's ops.

## What's new in Version 1.0.0 ##
+ Implemented a new framework that can be used to easily integrate different embedding algorithms to common DL frameworks.
+ Supports single-node & multi-node synchronized training with TensorFlow.
+ Integrated HugeCTR's DistributedSparseEmbedding algorithm.
+ Integrated All2AllDenseEmbedding algorithm.
+ Added custom Adam optimizer for SOK when TF version <= 2.4.

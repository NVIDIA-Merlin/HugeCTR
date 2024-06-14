# Get Started With SparseOperationKit #
This document will walk you through simple demos to get you familiar with SparseOperationKit (SOK).

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Importing SparseOperationKit](#importing-sparseoperationkit)
4. [Initializing SOK with TensorFlow and Horovod](#initializing-sok-with-tensorflow-and-horovod)
5. [Defining SOK Distributed Variables](#defining-sok-distributed-variables)
6. [Using SOK for Lookup](#using-sok-for-lookup)
7. [Performing Backward and Optimizer Update](#performing-backward-and-optimizer-update)
8. [Interaction Between SOK Variable and TensorFlow Tensor](#interaction-between-sok-variable-and-tensorflow-tensor)
9. [Dumping and Loading Indices and Weights](#dumping-and-loading-indices-and-weights)
10. [Incremental Dump of Keys and Values](#incremental-dump-of-keys-and-values)
11. [Additional Resources](#additional-resources)

## Introduction
SparseOperationKit (SOK) is a toolkit designed to facilitate the handling of sparse operations in TensorFlow, particularly for distributed training scenarios. It supports TensorFlow versions 1.15 and >=2.6 and integrates seamlessly with Horovod for communication.

## Installation
Refer to the [*Installation* section](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/intro_link.html#installation) to install SparseOperationKit on your system.

## Importing SparseOperationKit
To get started, import SparseOperationKit as follows:
```python
import sparse_operation_kit as sok
```
SOK automatically detects the TensorFlow version in use, ensuring a consistent interface regardless of the version.

## Initializing SOK with TensorFlow and Horovod ##

Currently, SOK uses Horovod for communication. Begin by importing Horovod and correctly binding a GPU to each process.For detailed instructions on binding GPUs with Horovod, please refer to the [Horovod with TensorFlow guide](https://horovod.readthedocs.io/en/stable/tensorflow.html#horovod-with-tensorflow):

```python
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

import sparse_operation_kit as sok


hvd.init()
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")  # nopep8

sok.init()
```

## Defining SOK Distributed Variables
SOK provides two types of distributed variables for storing sparse embedding weights: `sok.Variable` and `sok.DynamicVariable`.

### sok.Variable
`sok.Variable` is similar to `tf.Variable` but includes distributed methods. It supports two partition methods:
1. **Default Partition Method**: Distributes embedding IDs across GPUs in a round-robin manner.
1. **Localized Partition Method**: Assigns embedding tables to specific GPUs to reduce lookup communication overhead.

```python
# Default method of sok.Variable is Distributed method,
# If there are 2 GPUs in total, the shape of v1 on GPU0 will be [9, 3] and the shape
# on GPU1 will be [8, 3]
v1 = sok.Variable(np.arange(15 * 16).reshape(15, 16), dtype=tf.float32)

#If you want to assign a sok.Variable to a specific GPU, add the parameter mode=“localized:gpu_id” when defining sok.variable, where gpu_id refers to the rank number of a GPU in Horovod
v2 = sok.Variable(np.arange(15 * 16).reshape(15, 16), dtype=tf.float32,mode="localized:0")

print("v1:\n", v1)
print("v2:\n", v2)
```

### sok.DynamicVariable
`sok.DynamicVariable` uses a hash table as its backend and supports dynamic memory usage. It has two types of backends:
* [HierarchicalKV (HKV)](https://github.com/NVIDIA-Merlin/HierarchicalKV)
  * **HierarchicalKV** provides hierarchical key-value storage
  * It stores key-value pairs (feature-embedding) on high-bandwidth memory (HBM) of GPUs and in host memory.
   * HKV can provide an eviction feature to control the memory usage of the entire embedding table.
* [DynamicEmbeddingTable (DET)](https://github.com/NVIDIA-Merlin/HugeCTR/tree/main/sparse_operation_kit/kit_src/variable/impl/dynamic_embedding_table)
  * **DynamicEmbeddingTable** stores all key-value pairs in GPU memory
  * The lookup performance of DET is slightly better than HKV, but it lacks eviction functionality and cannot control the memory size of the embedding table.

Due to the more comprehensive features of HKV, it is the default backend for `sok.DynamicVariable`. If you want to use DET, you can specify the input parameter `var_type="hbm"` when declaring `sok.DynamicVariable`. Typically, DET is suitable to make quick prototypes and verify correctness because it requires fewer arguments when declaring.

Here is a code sample showing how to declare a `sok.DynamicVariable`:

```python
# To apply for different backends of dynamic embedding, you need to specify var_type when defining sok.DynamicVariable. 'hbm' corresponds to DET, and 'hybrid' corresponds to HKV.
v1 = sok.DynamicVariable(dimension=16,var_type="hbm", initializer="normal")

#init_capacity and max_capacity are parameters accepted by the HKV table. The meanings of these parameters can be found in the HKV documentation.
v2 = sok.DynamicVariable(
    dimension=16,
    var_type="hybrid",
    initializer="uniform",
    init_capacity=1024 * 1024,
    max_capacity=1024 * 1024,
    max_hbm_for_vectors=2,
)

print("v1:\n", v1)
print("v2:\n", v2)
```
As seen from the above example, using HKV as the backend for `sok.DynamicVariable` requires passing more arguments. These arguments are needed when creating the HKV hash table.
For details, you can refer to the arguments from [HKV Configuration Options]( https://github.com/NVIDIA-Merlin/HierarchicalKV?tab=readme-ov-file#configuration-options)

When not familiar with HKV arguments, it is recommended to set the following three arguments
- `init_capacity`: When initializing the HKV hash table, the number of key-values. This number must be a power of two due to HKV's limitations.
- `max_capacity`: The maximum number of key-value pairs the HKV hash table can grow to. After reaching this number, the HKV hash table will not grow any further. This number must be a power of two due to HKV's limitations.
- `max_hbm_for_vectors`: During the use of the HKV hash table, how much GPU memory can the values can occupy at most, in gigabytes.

## Using SOK for Lookup

SOK provides the `lookup_sparse` API for lookups, which accepts `tf.RaggedTensor` or `tf.SparseTensor` as indices. It can simultaneously lookup multiple instances of `sok.variable` by fusing their operations as a single one.

How to use the `sok.lookup_sparse` is consistent with `tf.nn.embedding_lookup_sparse`, except that it can accept the lists of  `sok.Variable`, `sp_id`, and `combiner`.

```python
indices1 = tf.SparseTensor(
    indices=[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], values=[1, 1, 3, 4, 5], dense_shape=[2, 3]
)
print("indices1:\n", indices1)
# indices1: batch_size=2, max_hotness=3
# [[1, 1]
#  [3, 4, 5]]

indices2 = tf.SparseTensor(
    indices=[[0, 0], [1, 0], [1, 1]], values=[1, 2, 3], dense_shape=[2, 2]
)

print("indices2:\n", indices2)
# indices2: batch_size=2, max_hotness=2
# [[1]
#  [2, 3]]

with tf.GradientTape() as tape:
    embeddings = sok.lookup_sparse(
        [v1, v2], [indices1, indices2], combiners=["sum", "mean"]
    )
    loss = 0.0
    for i, embedding in enumerate(embeddings):
        loss += tf.reduce_sum(embedding)
        print("embedding%d:\n" % (i + 1), embedding)
    # embedding1: [[6,  8,  10]
    #              [36, 39, 42]]
    # embedding2: [[5,  6,  7,  8,  9
    #              [25, 27, 29, 31, 33]]

```

## Performing Backward and Optimizer Update

For the backward process, `sok.lookup_sparse` behaves like otherTensorFlow operations.
However, raw TensorFlow optimizers cannot apply gradients to `sok.DynamicVariable`. Use `sok.OptimizerWrapper` to wrap the optimizer.

```python
#define a tf optimizer , and then warp it to sok_optimizer, then can use it on sok.DynamicVariable`
optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
sok_optimizer = sok.OptimizerWrapper(optimizer)
grads = tape.gradient(loss, [v1, v2])
for i, grad in enumerate(grads):
    print("grad%d:\n" % (i + 1), grad)

sok_optimizer.apply_gradients(zip(grads, [v1, v2]))
print("v1:\n", v1)
print("v2:\n", v2)
```

## Interaction Between SOK Variable and TensorFlow Tensor

SOK provides `sok.export` and `sok.assign` APIs for interaction between `sok.Variable` and `TensorFlow.tensor`.

**Note**: since SOKvariables are distributed across multiple GPUs, the `sok.export` and `sok.assign` APIs will only export and assign the portion of the data on the local GPU. This means that those operations do not handle the global size of the variables but rather focus on the local segments specific to each GPU.

```python
#Generate the keys and values you want to assign to sok.Variable
with tf.device("CPU"):
    indices = tf.convert_to_tensor([i for i in range(24)], dtype=tf.int64)
    values = tf.convert_to_tensor(np.random.rand(24, 16), dtype=tf.float32)
sok.assign(v1, indices, values)

#Export the keys and values of sok.Variable to two tf.tensor placed on the CPU.
ex_v1_indices, ex_v1_values = sok.export(v1)
print("ex_v1_indices:\n", ex_v1_indices)
print("ex_v1_values:\n", ex_v1_values)
```

## Dumping and Loading Indices and Weights

SOK provides `sok.dump` and `sok.load` for dumping/loading trained keys and values to/from the filesystem.
- `sok.dump` dumps the keys, values, and optimizer states as multiple binary files with a simple file header.
- `sok.load` loads the keys, values, and optimizer states, from the binary files created by `sok.dump`, automatically distributing them to the GPUs."

**Note**:optimizer states are optional. If they are unspecified in calling the APIs above, only the keys and values are loaded.

```python
path = "./weights"
sok_vars = [v1,v2]
sok.dump("./weight", sok_vars, sok_optimizer)
sok.load("./weight", sok_vars, sok_optimizer)
```

## Incremental Dump of Keys and Values

SOK supports incremental dumps, allowing you to dump keys and values updated after a specific time threshold (in UTC) into a Numpy array.

```python
import pytz
from datetime import datetime

#should convert datatime to utc time
utc_time_threshold = datetime.now(pytz.utc)

#####
#Need do some lookup forward and backward or sok.assign
#####

sok_vars = [v1,v2]
#keys and values are Numpy array
keys, values = sok.incremental_model_dump(sok_vars, utc_time_threshold)
```

## Additional Resources
For more examples and API descriptions, see the [Example section](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/examples/examples.html) and [API section](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/index.html).

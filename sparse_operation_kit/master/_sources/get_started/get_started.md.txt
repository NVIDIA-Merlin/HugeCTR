# Get Started With SparseOperationKit #
This document will walk you through simple demos to get you familiar with SparseOperationKit.

<div class="admonition note">
<p class="admonition-title">See also</p>
<p>For experts or more examples, please refer to Examples section</p>
</div>

Refer to the [*Installation* section](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/intro_link.html#installation) to install SparseOperationKit on your system.

## Import SparseOperationKit ##
```python
import sparse_operation_kit as sok
```
The SOK supports the TensorFlow 1.15 and >=2.6. It automatically detects the TensorFlow version in use on behalf of users. The SOK interface is identical regardless of which TensorFlow version is used.

## Use SOK with TensorFLow ##

Currently, we use horovod for communication. So in the beginning, you need to import horovod and correctly bind a GPU to each process like this:

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

Next, in order to use the distributed embedding op, you need to create a variable on each process that represents a portion of the entire embedding table, whose shape is also a subset of the full embedding table. We provide a tensorflow variable wrapper to help you simplify this process.

```python
# Default mode of sok.Variable is Distributed mode
# If there are 2 GPUs in total, the shape of v1 on GPU0 will be [9, 3] and the shape
# on GPU1 will be [8, 3]
v1 = sok.Variable(np.arange(17 * 3).reshape(17, 3), dtype=tf.float32)
v2 = sok.Variable(np.arange(7 * 5).reshape(7, 5), dtype=tf.float32)
print("v1:\n", v1)
print("v2:\n", v2)
```

Then, create the indices for the embedding lookup. This step is no different from the normal tensorflow.
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
```

Then, use sok's embedding op to do the lookup. Note that here we pass two embedding variables and two indices into the lookup at the same time through a list, this fused operation will bring performance gain for us.
```python
with tf.GradientTape() as tape:
    embeddings = sok.lookup_sparse(
        [v1, v2], [indices1, indices2], combiners=["sum", "sum"]
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

Finally, update the variable like normal tensorflow.

```python
# If there are 2 GPUs in total
# GPU0:
#   In Distributed mode: shape of grad of v1 will be [1, 3], shape of grad of v2 will be [1, 5]
#   In Localized mode: shape of grad of v1 will be [4, 3], grad of v2 will None
# GPU1:
#   In Distributed mode: shape of grad of v1 will be [3, 3], shape of grad of v2 will be [2, 5]
#   In Localized mode: grad of v1 will be None, shape of grad of v2 will be [3, 5]
grads = tape.gradient(loss, [v1, v2])
for i, grad in enumerate(grads):
    print("grad%d:\n" % (i + 1), grad)

# Use tf.keras.optimizer to optimize the sok.Variable
optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
optimizer.apply_gradients(zip(grads, [v1, v2]))
print("v1:\n", v1)
print("v2:\n", v2)
```

For more examples and API descriptions see the Example section and API section.

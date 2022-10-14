# Get Started With SparseOperationKit #
This document will walk you through simple demos to get you familiar with SparseOperationKit.

<div class="admonition note">
<p class="admonition-title">See also</p>
<p>For experts or more examples, please refer to Examples section</p>
</div>

<div class="admonition Important">
<p class="admonition-title">Important</p>
<p>We strongly recommend using the new version of SOK under `sok.experiment`. After this new version is stable, the old SOK will be deprecated. See the Experimental Features section on this page to get started with it.</p>
</div>

Refer to the [*Installation* section](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/intro_link.html#installation) to install SparseOperationKit on your system.

## Import SparseOperationKit ##
```python
import sparse_operation_kit as sok
```
SOK supports TensorFlow 1.15 and 2.x, and automatically detects the version of TensorFlow from your program. The SOK API signatures for TensorFlow 2.x and TensorFlow 1.15 are identical.

## TensorFlow 2.x ##

### Define a model with TensorFlow ###
The structure of this demo model is depicted in Fig 1.

<br><img src=../images/demo_model_structure.png></br>
<center><b>Fig 1. The structure of demo model</b></center>
<br>

To define the model, you can use either use *subclassing* or the *functional API*.

**Subclassing approach**. The following code sample shows how this demo model can be created by subclassing `tf.keras.Model`. Additional information about `tf.keras.Model` and its customization options is available [here](https://tensorflow.google.cn/guide/keras/custom_layers_and_models).

```python
import tensorflow as tf

class DemoModel(tf.keras.models.Model):
    def __init__(self,
                 max_vocabulary_size_per_gpu,
                 slot_num,
                 nnz_per_slot,
                 embedding_vector_size,
                 num_of_dense_layers,
                 **kwargs):
        super(DemoModel, self).__init__(**kwargs)

        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.slot_num = slot_num            # the number of feature-fileds per sample
        self.nnz_per_slot = nnz_per_slot    # the number of valid keys per feature-filed
        self.embedding_vector_size = embedding_vector_size
        self.num_of_dense_layers = num_of_dense_layers

        # this embedding layer will concatenate each key's embedding vector
        self.embedding_layer = sok.All2AllDenseEmbedding(
                    max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                    embedding_vec_size=self.embedding_vector_size,
                    slot_num=self.slot_num,
                    nnz_per_slot=self.nnz_per_slot)

        self.dense_layers = list()
        for _ in range(self.num_of_dense_layers):
            self.layer = tf.keras.layers.Dense(units=1024, activation="relu")
            self.dense_layers.append(self.layer)

        self.out_layer = tf.keras.layers.Dense(units=1, activation=None)

    def call(self, inputs, training=True):
        # its shape is [batchsize, slot_num, nnz_per_slot, embedding_vector_size]
        emb_vector = self.embedding_layer(inputs, training=training)

        # reshape this tensor, so that it can be processed by Dense layer
        emb_vector = tf.reshape(emb_vector, shape=[-1, self.slot_num * self.nnz_per_slot * self.embedding_vector_size])

        hidden = emb_vector
        for layer in self.dense_layers:
            hidden = layer(hidden)

        logit = self.out_layer(hidden)
        return logit
```

**Functional API approach**. The following code sample shows how to create a model with the `TensorFlow functional API`. For information about the API, see the [TensorFlow functional API](https://tensorflow.google.cn/guide/keras/functional).

```python
import tensorflow as tf

def create_DemoModel(max_vocabulary_size_per_gpu,
                     slot_num,
                     nnz_per_slot,
                     embedding_vector_size,
                     num_of_dense_layers):
    # config the placeholder for embedding layer
    input_tensor = tf.keras.Input(
                type_spec=tf.TensorSpec(shape=(None, slot_num, nnz_per_slot),
                dtype=tf.int64))

    # create embedding layer and produce embedding vector
    embedding_layer = sok.All2AllDenseEmbedding(
                max_vocabulary_size_per_gpu=max_vocabulary_size_per_gpu,
                embedding_vec_size=embedding_vector_size,
                slot_num=slot_num,
                nnz_per_slot=nnz_per_slot)
    embedding = embedding_layer(input_tensor)

    # create dense layers and produce logit
    embedding = tf.keras.layers.Reshape(
                target_shape=(slot_num * nnz_per_slot * embedding_vector_size,))(embedding)

    hidden = embedding
    for _ in range(num_of_dense_layers):
        hidden = tf.keras.layers.Dense(units=1024, activation="relu")(hidden)
    logit = tf.keras.layers.Dense(units=1, activation=None)

    model = tf.keras.Model(inputs=input_tensor, outputs=logit)
    return model
```

### Use SparseOperationKit with tf.distribute.Strategy ###
SparseOperationKit is compatible with `tf.distribute.Strategy`. More specificly, `tf.distribute.MirroredStrategy` and `tf.distribute.MultiWorkerMirroredStrategy`.

#### with tf.distribute.MirroredStrategy ####
The `tf.distribute.MirroredStrategy` class enables data-parallel synchronized training on a machine with multiple GPUs. For more information, see the TensorFlow documentation for the [MirroredStrategy](https://tensorflow.google.cn/api_docs/python/tf/distribute/MirroredStrategy) class.

<div class="admonition Caution">
<p class="admonition-title">Caution</p>
<p>The programming model for MirroredStrategy is single-process & multiple-threads. CPython is prone to Global Interpreter Lock (GIL). GIL makes it hard to fully leverage all available CPU cores, which might impact the end-to-end training / inference performance. Therefore, the MirroredStrategy is not recommended for synchronized training using multiple GPUs.</p>
</div>

***create MirroredStrategy***
```python
strategy = tf.distribute.MirroredStrategy()
```
<div class="admonition Tip">
<p class="admonition-title">Tip</p>
<p>By default, MirroredStrategy will use all available GPUs in one machine. You can select which GPUs should be used for synchronized training by specifying either `CUDA_VISIBLE_DEVICES` or `tf.config.set_visible_devices`.</p>
</div>

***create model instance under MirroredStrategy.scope***
```python
global_batch_size = 65536
use_tf_opt = True

with strategy.scope():
    sok.Init(global_batch_size=global_batch_size)

    model = DemoModel(
        max_vocabulary_size_per_gpu=1024,
        slot_num=10,
        nnz_per_slot=5,
        embedding_vector_size=16,
        num_of_dense_layers=7)

    if not use_tf_opt:
        emb_opt = sok.optimizers.Adam(learning_rate=0.1)
    else:
        emb_opt = tf.keras.optimizers.Adam(learning_rate=0.1)

    dense_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
```
Prior to using a DNN model that is built with SOK, you must call `sok.Init` to initalize SOK. Please refer to its [API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/init.html#module-sparse_operation_kit.core.initialize) for further information.

***define training step***
```python
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
def _replica_loss(labels, logits):
    loss = loss_fn(labels, logits)
    return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

@tf.function
def _train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = _replica_loss(lables, logits)
    emb_var, other_var = sok.split_embedding_variable_from_others(model.trainable_variables)
    grads, emb_grads = tape.gradient(loss, [other_var, emb_var])
    if use_tf_opt:
        with sok.OptimizerScope(emb_var):
            emb_opt.apply_gradients(zip(emb_grads, emb_var),
                                    experimental_aggregate_gradients=False)
    else:
        emb_opt.apply_gradients(zip(emb_grads, emb_var),
                                experimental_aggregate_gradients=False)
    dense_opt.apply_gradients(zip(grads, other_var))
    return loss
```

If you are using native TensorFlow optimizers, such as `tf.keras.optimizers.Adam`, then `sok.OptimizerScope` must be used. Please refer to its [API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/utils/opt_scope.html#sparseoperationkit-optimizer-scope) for further information.

***start training***
```python
dataset = ...

for i, (inputs, labels) in enumerate(dataset):
    replica_loss = strategy.run(_train_step, args=(inputs, labels))
    total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, replica_loss, axis=None)
    print("[SOK INFO]: Iteration: {}, loss: {}".format(i, total_loss))
```

After these steps, the `DemoModel` will be successfully trained.

#### With tf.distribute.MultiWorkerMirroredStrategy ####
`tf.distribute.MultiWorkerMirroredStrategy` allows data-parallel synchronized training across multiple machines with multiple GPUs in each machine. Its [documentation](https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy) can be found [here](https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy).

<div class="admonition Caution">
<p class="admonition-title">Caution</p>
<p>The programming model for the MultiWorkerMirroredStrategy is multiple processes plus multi-threading. Hence, each process owns multiple threads to control the indidvidual GPUs in each machine. GILs in the CPython interpreter can make it hard to fully leverage all available CPU cores in each machine, which might impact the end-to-end training / inference performance. Therefore, it is recommended to use multiple processes in each machine, and each process controls one GPU.</p>
</div>

<div class="admonition Important">
<p class="admonition-title">Important</p>
<p>By default, MultiWorkerMirroredStrategy will use all available GPUs in each process. You can limit GPU access for each process by setting either `CUDA_VISIBLE_DEVICES` or `tf.config.set_visible_devices`.</p>
</div>

***create MultiWorkerMirroredStrategy***
```python
import os, json

worker_num = 8 # how many GPUs are used
task_id = 0    # this process controls which GPU

os.environ["CUDA_VISIBLE_DEVICES"] = str(task_id) # this procecss only controls this GPU

port = 12345 # could be arbitrary unused port on this machine
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {"worker": ["localhost:" + str(port + i)
                            for i in range(worker_num)]},
    "task": {"type": "worker", "index": task_id}
})
strategy = tf.distribute.MultiWorkerMirroredStrategy()
```

***Other Steps***<br>
The steps ***create model instance under MultiWorkerMirroredStrategy.scope***, ***define training step*** and ***start training*** are the same as those described in [with tf.distribute.MirroredStrategy](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/get_started/get_started.html#with-tf-distribute-mirroredstrategy). Please check that section.

***launch training program***<br>
Because multiple CPU processes are used in each machine for synchronized training, MPI can be used to launch this program. For example using:
```shell
$ mpiexec -np 8 [mpi-args] python3 main.py [python-args]
```

### Use SparseOperationKit with Horovod ###
SparseOperationKit is also compatible with [Horovod](https://horovod.ai), which is similar to `tf.distribute.MultiWorkerMirroredStrategy`.

***initialize horovod for tensorflow***
```python
import horovod.tensorflow as hvd
hvd.init()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank()) # this process only controls one GPU
```

***create model instance***
```python
global_batch_size = 65536
use_tf_opt = True

sok.Init(global_batch_size=global_batch_size)

model = DemoModel(max_vocabulary_size_per_gpu=1024,
                  slot_num=10,
                  nnz_per_slot=5,
                  embedding_vector_size=16,
                  num_of_dense_layers=7)

if not use_tf_opt:
    emb_opt = sok.optimizers.Adam(learning_rate=0.1)
else:
    emb_opt = tf.keras.optimizers.Adam(learning_rate=0.1)

dense_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
```
Prior to using a DNN model built with SOK, `sok.Init` must be called to perform certain initilization steps. Please refer to its [API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/init.html#module-sparse_operation_kit.core.initialize) for further information.

***define training step***
```python
loss_fn = tf.keras.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
def _replica_loss(labels, logits):
    loss = loss_fn(labels, logits)
    return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

@tf.function
def _train_step(inputs, labels, first_batch):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = _replica_loss(labels, logits)
    emb_var, other_var = sok.split_embedding_variable_from_others(model.trainable_variables)
    emb_grads, other_grads = tape.gradient(loss, [emb_var, other_var])
    if use_tf_opt:
        with sok.OptimizerScope(emb_var):
            emb_opt.apply_gradients(zip(emb_grads, emb_var),
                                    experimental_aggregate_gradients=False)
    else:
        emb_opt.apply_gradients(zip(emb_grads, emb_var),
                                experimental_aggregate_gradients=False)

    other_grads = [hvd.allreduce(grads) for grads in other_grads]
    dense_opt.apply_gradients(zip(other_grads, other_var))

    if first_batch:
        hvd.broadcast_variables(other_var, root_rank=0)
        hvd.broadcast_variables(dense_opt.variables(), root_rank=0)

    return loss
```
If you use native TensorFlow optimizers, such as `tf.keras.optimizers.Adam`, then `sok.OptimizerScope` must be used. Please see its [API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/utils/opt_scope.html#sparseoperationkit-optimizer-scope) for further information.

***start training***
```python
dataset = ...

for i, (inputs, labels) in enumerate(dataset):
    replica_loss = _train_step(inputs, labels, 0 == i)
    total_loss = hvd.allreduce(replica_loss)
    print("[SOK INFO]: Iteration: {}, loss: {}".format(i, total_loss))
```

***launch training program***<br>
You can use `horovodrun` or `mpiexec` to launch multiple processes in each machine for synchronized training. For example:
```shell
$ horovodrun -np 8 -H localhost:8 python3 main.py [python-args]
```

## TensorFlow 1.15 ##
SOK is compatible with TensorFlow 1.15. But due to some restrictions in TF 1.15, only Horovod can be used as the communication protocol.

### Using SparseOperationKit with Horovod ###
***initialize horovod for tensorflow***
```python
import horovod.tensorflow as hvd
hvd.init()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank()) # this process only controls one GPU
```

***create model instance***
```python
global_batch_size = 65536
use_tf_opt = True

sok_init_op = sok.Init(global_batch_size=global_batch_size)

model = DemoModel(max_vocabulary_size_per_gpu=1024,
                  slot_num=10,
                  nnz_per_slot=5,
                  embedding_vector_size=16,
                  num_of_dense_layers=7)

if not use_tf_opt:
    emb_opt = sok.optimizers.Adam(learning_rate=0.1)
else:
    emb_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
dense_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
```
Prior to using a DNN model built with SOK, `sok.Init` must be called to perform certain initilization steps. Please refer to its [API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/init.html#module-sparse_operation_kit.core.initialize) for further information.

***define training step***
```python
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
def _replica_loss(labels, logits):
    loss = loss_fn(labels, logits)
    return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

def train_step(inputs, labels, training):
    logits = model(inputs, training=training)
    loss = _replica_loss(labels, logit)
    emb_var, other_var = sok.split_embedding_variable_from_others(model.trainable_variables)
    grads = tf.gradients(loss, emb_var + other_var, colocate_gradients_with_ops=True)
    emb_grads, other_grads = grads[:len(emb_var)], grads[len(emb_var):]

    if use_tf_opt:
        with sok.OptimizerScope(emb_var):
            emb_train_op = emb_opt.apply_gradients(zip(emb_grads, emb_var))
    else:
        emb_train_op = emb_opt.apply_gradients(zip(emb_grads, emb_var))

    other_grads = [hvd.allreduce(grad) for grad in other_grads]
    other_train_op = dense_opt.apply_gradients(zip(other_grads, other_var))

    with tf.control_dependencies([emb_train_op, other_train_op]):
        total_loss = hvd.reduce(loss)
        total_loss = tf.identity(total_loss)

        return total_loss
```
If you are using native TensorFlow optimizers, such as `tf.keras.optimizers.Adam`, then `sok.OptimizerScope` must be used. Please see its [API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/utils/opt_scope.html#sparseoperationkit-optimizer-scope) for further information.

***start training***
```python
dataset = ...

loss = train_step(inputs, labels)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(sok_init_op)
    sess.run(init_op)

    for step in range(iterations):
        loss_v = sess.run(loss)
        print("[SOK INFO]: Iteration: {}, loss: {}".format(step, loss_v))
```
**Please be noted that `sok_init_op` must be the first step in `sess.run`, even before variables initialization.**

***launch training program***
You can use `horovodrun` or `mpiexec` to launch multiple processes in each machine for synchronized training. For example:
```shell
$ horovodrun -np 8 -H localhost:8 python main.py [args]
```

## Experimental Features

Currently, we use horovod for communication. So in the beginning, you need to import horovod and correctly bind a GPU to each process like this:

```python
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from sparse_operation_kit import experiment as sok


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
        [v1, v2], [indices1, indices2], hotness=[3, 2], combiners=["sum", "sum"]
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

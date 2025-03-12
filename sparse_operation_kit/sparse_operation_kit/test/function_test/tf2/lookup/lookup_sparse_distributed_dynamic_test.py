"""
 Copyright (c) 2022, NVIDIA CORPORATION.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

import sparse_operation_kit as sok


if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    sok.init()

    rows = [8192 * 10, 8192]
    cols = [128, 4]
    hotness = [10, 3]
    combiners = ["sum", "sum"]
    batch_size = 8192
    iters = 100
    initial_vals = [13, 17]

    # sok variables
    sok_vars = [
        sok.DynamicVariable(dimension=cols[i], initializer=str(initial_vals[i]), var_type="hbm")
        for i in range(len(cols))
    ]
    local_indices = []
    for row in rows:
        local_size = row // hvd.size()
        if hvd.rank() < row % hvd.size():
            local_size += 1
        indices = np.arange(local_size) * hvd.size() + hvd.rank()
        indices = tf.convert_to_tensor(indices, dtype=tf.int64)
        local_indices.append(indices)

    # indices
    total_indices = []
    for i in range(len(rows)):
        offsets = np.random.randint(1, hotness[i] + 1, iters * batch_size)
        offsets = tf.convert_to_tensor(offsets, dtype=tf.int64)
        offsets = hvd.broadcast(offsets, root_rank=0)
        values = np.random.randint(0, rows[i], tf.reduce_sum(offsets))
        values = tf.convert_to_tensor(values, dtype=tf.int64)
        values = hvd.broadcast(values, root_rank=0)
        total_indices.append(tf.RaggedTensor.from_row_lengths(values, offsets))
    left = batch_size // hvd.size() * hvd.rank()
    right = batch_size // hvd.size() * (hvd.rank() + 1)

    # initialize optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

    if sok.tf_version[0] == 2 and sok.tf_version[1] >= 17:
        import tf_keras as tfk

        optimizer_for_sok = tfk.optimizers.legacy.SGD(learning_rate=1.0)
    else:
        optimizer_for_sok = tf.keras.optimizers.SGD(learning_rate=1.0)
    sok_optimizer = sok.OptimizerWrapper(optimizer_for_sok)

    tf_vars = [
        tf.Variable(tf.constant(initial_vals[i], shape=[rows[i], cols[i]], dtype=tf.float32))
        for i in range(len(rows))
    ]

    @tf.function
    def step(params, indices):
        with tf.GradientTape() as tape:
            embeddings = sok.lookup_sparse(params, indices, combiners=combiners)
            loss = 0
            for i in range(len(embeddings)):
                loss = loss + tf.reduce_sum(embeddings[i])
        grads = tape.gradient(loss, params)
        sok_optimizer.apply_gradients(zip(grads, params))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss

    # Do training
    loss1 = []
    ts = []
    t = time.time()
    for i in range(iters):
        ts.append(time.time() - t)
        t = time.time()
        indices = []
        for j in range(len(total_indices)):
            indices.append(total_indices[j][i * batch_size + left : i * batch_size + right])
        loss = step(sok_vars, indices)
        loss1.append(loss)
        print("-" * 30 + "iteration %d" % i + "-" * 30)
        print("loss:", loss)
    out1 = []
    for i in range(len(sok_vars)):
        out1.append(tf.nn.embedding_lookup(sok_vars[i], local_indices[i]))

    @tf.function
    def step2(params, indices):
        with tf.GradientTape() as tape:
            loss = 0
            for i in range(len(params)):
                embedding = tf.nn.embedding_lookup_sparse(
                    params[i], indices[i], None, combiner=combiners[i]
                )
                loss = loss + tf.reduce_sum(embedding)
        grads = tape.gradient(loss, params)
        grads = [hvd.allreduce(grad, op=hvd.Sum) for grad in grads]
        optimizer.apply_gradients(zip(grads, params))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss

    loss2 = []
    for i in range(iters):
        indices = []
        for j in range(len(total_indices)):
            indices.append(
                total_indices[j][i * batch_size + left : i * batch_size + right].to_sparse()
            )
        loss = step2(tf_vars, indices)
        loss2.append(loss)
        print("-" * 30 + "iteration %d" % i + "-" * 30)
        print("tf loss:", loss)
    out2 = []
    for i, v in enumerate(tf_vars):
        out2.append(tf.nn.embedding_lookup(v, local_indices[i]))

    # Check results
    diff = 0
    for i in range(len(out1)):
        length = out1[i] ** 2 + out2[i] ** 2 + 1e-8
        diff = diff + tf.reduce_sum((out1[i] - out2[i]) ** 2 / length)
    print("[SOK INFO] diff:", diff)
    assert diff < 1e-6

    diff = 0
    for i in range(iters):
        length = loss1[i] ** 2 + loss2[i] ** 2 + 1e-8
        diff = diff + (loss1[i] - loss2[i]) ** 2 / length
    print("[SOK INFO] loss diff:", diff)
    assert diff < 1e-6

    print("[SOK INFO] lookup_sparse distributed with dynamic variable test passed")
    ts = ts[5:]
    print("[SOK INFO] Average time: %f ms/iteration" % (sum(ts) / len(ts) * 1000))

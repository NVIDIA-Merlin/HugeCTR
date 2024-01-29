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
import pytz
from datetime import datetime
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
    iters = 5
    initial_vals = [13, 17]
    sleep_seconds = 15

    # sok variables
    sok_vars = [
        sok.DynamicVariable(
            dimension=cols[i],
            var_type="hybrid",
            initializer=str(initial_vals[i]),
            init_capacity=1024 * 1024,
            max_capacity=1024 * 1024,
        )
        for i in range(len(cols))
    ]
    print("HKV var created")
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
    optimizer = tf.optimizers.SGD(learning_rate=1.0, momentum=0.9)
    # sok_optimizer = sok.SGD(lr=1.0)
    sok_optimizer = sok.OptimizerWrapper(optimizer)

    def step(params, indices):
        with tf.GradientTape() as tape:
            embeddings = sok.lookup_sparse(params, indices, combiners=combiners)
            loss = 0
            for i in range(len(embeddings)):
                loss = loss + tf.reduce_sum(embeddings[i])
        grads = tape.gradient(loss, params)
        sok_optimizer.apply_gradients(zip(grads, params))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss, embeddings

    indices_records = []
    time_records = []
    for i in range(iters):
        indices = []
        indices_global = []
        for j in range(len(total_indices)):
            indices.append(total_indices[j][i * batch_size + left : i * batch_size + right])
            indices_global.append(total_indices[j][i * batch_size : (i + 1) * batch_size])
        time.sleep(sleep_seconds)
        loss, embeddings = step(sok_vars, indices)
        indices_records.append(indices_global)
        time.sleep(sleep_seconds)
        utc_time = datetime.now(pytz.utc)
        time_records.append(utc_time)
        if i > 0:
            time_before = time_records[i - 1]
            keys, values = sok.incremental_model_dump(sok_vars, time_before)
            num_lookups = len(keys)
            indices_before = indices_records[i]
            for lookup_id in range(num_lookups):
                indices_flat_before = indices_before[lookup_id].flat_values
                indices_np = indices_flat_before.numpy()
                indices_np, unique_reverse_indices = np.unique(indices_np, return_index=True)
                indices_np = np.sort(indices_np)
                tmp_keys = keys[lookup_id]
                tmp_keys = np.sort(tmp_keys)
                np.testing.assert_array_equal(indices_np, tmp_keys)
            print("____________iter {} is pass!________________".format(str(i)))

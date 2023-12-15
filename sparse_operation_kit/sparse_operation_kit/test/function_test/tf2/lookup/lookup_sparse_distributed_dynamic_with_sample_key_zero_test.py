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
    combiners = ["sum", "mean"]
    batch_size = 8192
    iters = 1
    initial_vals = [13, 17]

    # sok variables
    sok_vars = [
        sok.DynamicVariable(dimension=cols[i], initializer=str(initial_vals[i]))
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
    is_zeros_list = []
    total_indices = []
    for i in range(len(rows)):
        offsets = np.random.randint(0, hotness[i] + 1, iters * batch_size)
        is_zero = np.where(offsets == 0)[0]
        is_zeros_list.append(is_zero)
        offsets = tf.convert_to_tensor(offsets, dtype=tf.int64)
        offsets = hvd.broadcast(offsets, root_rank=0)
        values = np.random.randint(0, rows[i], tf.reduce_sum(offsets))
        values = tf.convert_to_tensor(values, dtype=tf.int64)
        values = hvd.broadcast(values, root_rank=0)
        total_indices.append(tf.RaggedTensor.from_row_lengths(values, offsets))
    left = batch_size // hvd.size() * hvd.rank()
    right = batch_size // hvd.size() * (hvd.rank() + 1)

    @tf.function
    def step(params, indices):
        embeddings = sok.lookup_sparse(params, indices, combiners=combiners)
        return embeddings

    indices = []
    # Do lookup
    for i in range(len(total_indices)):
        indices.append(total_indices[i][left:right])
    embeddings = step(sok_vars, indices)
    embedding_np = [embedding.numpy() for embedding in embeddings]
    for i in range(len(embedding_np)):
        tmp_emb = embedding_np[i]
        tmp_zeros = tmp_emb[is_zeros_list[i], :]
        assert (np.abs(tmp_zeros - 0) < 1e-6).all()

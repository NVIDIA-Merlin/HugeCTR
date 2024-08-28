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

np.set_printoptions(threshold=np.inf)

if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    sok.init()

    rows = [8192 * 2048, 8192 * 8192]
    cols = [128, 4]
    hotness = [1, 1]
    combiners = ["sum", "mean"]
    batch_size = 8192
    iters = 1
    filter_iters = 5
    initial_vals = [13, 17]

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

    # indices
    total_indices = []
    total_indices_np = []
    for i in range(len(rows)):
        offsets = np.random.randint(1, hotness[i] + 1, iters * batch_size)
        offsets = tf.convert_to_tensor(offsets, dtype=tf.int64)
        offsets = hvd.broadcast(offsets, root_rank=0)
        values = np.random.randint(0, rows[i], tf.reduce_sum(offsets))
        values = tf.convert_to_tensor(values, dtype=tf.int64)
        values = hvd.broadcast(values, root_rank=0)
        total_indices_np.append(values)
        total_indices.append(tf.RaggedTensor.from_row_lengths(values, offsets))
    left = batch_size // hvd.size() * hvd.rank()
    right = batch_size // hvd.size() * (hvd.rank() + 1)

    unique_indices = []
    for i in range(len(total_indices_np)):
        unique_indices.append(np.unique(total_indices_np[i]))

    # initialize optimizer
    optimizer = tf.optimizers.SGD(learning_rate=1.0, momentum=0.9)
    sok_optimizer = sok.OptimizerWrapper(optimizer)

    def step(params, indices, use_filter=False):
        with tf.GradientTape() as tape:
            embeddings = sok.lookup_sparse(
                params, indices, combiners=combiners, use_low_frequency_filter=use_filter
            )
            loss = 0
            for i in range(len(embeddings)):
                loss = loss + tf.reduce_sum(embeddings[i])
        grads = tape.gradient(loss, params)
        sok_optimizer.apply_gradients(zip(grads, params))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss, embeddings

    indices_records = []
    for i in range(iters):
        loss, embeddings = step(sok_vars, total_indices)
    print("____________pre lookup is done!________________".format(str(i)))

    # indices
    total_indices_filter = []
    total_indices_filter_np = []
    for i in range(len(rows)):
        offsets = np.random.randint(1, hotness[i] + 1, filter_iters * batch_size)
        offsets = tf.convert_to_tensor(offsets, dtype=tf.int64)
        offsets = hvd.broadcast(offsets, root_rank=0)
        values = np.random.randint(0, rows[i], tf.reduce_sum(offsets))
        values = tf.convert_to_tensor(values, dtype=tf.int64)
        values = hvd.broadcast(values, root_rank=0)
        total_indices_filter_np.append(values)
        total_indices_filter.append(tf.RaggedTensor.from_row_lengths(values, offsets))

    left = batch_size // hvd.size() * hvd.rank()
    right = batch_size // hvd.size() * (hvd.rank() + 1)

    def check_zero_line(arr):
        zero_rows = [idx for idx, row in enumerate(arr) if np.all(row == 0)]
        rows = arr.shape[0]
        if rows == 0:
            return 0, 0, 0
        zero_count = 0
        for i in range(rows):
            tmp_line = arr[i, :]
            if np.all(tmp_line == 0):
                zero_count += 1
        return zero_count, rows, zero_count / rows

    for i in range(iters):
        indices = []
        indices_np = []
        indices_new_np = []
        masks = []
        for j in range(len(total_indices_filter)):
            tmp_indices_tensor = total_indices_filter[j][
                i * batch_size + left : i * batch_size + right
            ]
            indices.append(tmp_indices_tensor)
            indices_np.append(np.squeeze(tmp_indices_tensor.numpy()))
            mask = np.isin(indices_np[j], unique_indices[j])
            masks.append(mask)
        loss, embeddings = step(sok_vars, indices, use_filter=True)
        for k, embedding in enumerate(embeddings):
            embedding_np = embedding.numpy()
            mask_no_filter_index = np.where(
                masks[k] == True,
            )[0]
            mask_filter_index = np.where(
                masks[k] == False,
            )[0]

            print("mask_no_filter_index = ", mask_no_filter_index)
            print("mask_filter_index = ", mask_filter_index)
            embedding_no_filter_np = embedding_np[mask_no_filter_index, :]
            embedding_filter_np = embedding_np[mask_filter_index, :]

            print("embedding_no_filter_np = ", embedding_no_filter_np)
            print("embedding_filter_np = ", embedding_filter_np)
            print("embedding_np = ", embedding_np.shape)
            print("mask = ", mask.shape)
            zero_count_no_filter, _, zero_rate_no_filter = check_zero_line(embedding_no_filter_np)
            zero_count_filter, _, zero_rate_filter = check_zero_line(embedding_filter_np)
            print("zero_rate_no_filter = ", zero_rate_no_filter)
            print("zero_rate_filter = ", zero_rate_filter)
            assert zero_count_filter >= 0
            assert zero_count_no_filter == 0
            print("low frequency filter is pass")

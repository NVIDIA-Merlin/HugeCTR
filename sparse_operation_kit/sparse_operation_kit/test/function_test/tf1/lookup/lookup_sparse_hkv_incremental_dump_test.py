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
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sok.init()

    gpu_num = hvd.size()
    rows = [8192] * gpu_num
    cols = [128 - 8 * x for x in range(gpu_num)]
    hotness = [1 for x in range(gpu_num)]
    combiners = ["mean"] * np.floor(gpu_num / 2).astype(np.int32) + ["sum"] * np.ceil(
        gpu_num - gpu_num / 2
    ).astype(np.int32)

    batch_size = 8192
    iters = 5
    initial_vals = [3 + x for x in range(gpu_num)]
    sleep_seconds = 15
    gpus = np.arange(gpu_num)
    rank = hvd.rank()

    # sok variables
    sok_vars = []
    if len(gpus) >= 2:
        for i in range(len(cols)):
            v = sok.DynamicVariable(
                dimension=cols[i],
                var_type="hybrid",
                initializer=str(initial_vals[i]),
                init_capacity=1024 * 1024,
                max_capacity=1024 * 1024,
            )
            sok_vars.append(v)
    else:
        for i in range(len(cols)):
            v = sok.DynamicVariable(
                dimension=cols[i],
                var_type="hybrid",
                initializer=str(initial_vals[i]),
                init_capacity=1024 * 1024,
                max_capacity=1024 * 1024,
            )
            sok_vars.append(v)

    local_indices = []
    local_indices_numpy = []

    for i, row in enumerate(rows):
        if rank == gpus[i]:
            indices_np = np.arange(row)
            local_indices_numpy.append(indices_np)
            indices = tf.placeholder(shape=[None], dtype=tf.int64)
            local_indices.append(indices)
        else:
            local_indices_numpy.append(None)
            local_indices.append(None)

    # indices
    offsets_numpy = []
    values_numpy = []
    offsets = []
    values = []
    value_tensors = []
    total_indices = []
    for i in range(len(rows)):
        offset_np = np.random.randint(1, hotness[i] + 1, iters * batch_size)
        offsets_numpy.append(offset_np)
        offset = tf.placeholder(shape=[None], dtype=tf.int64)
        offset = hvd.broadcast(offset, root_rank=0)
        offsets.append(offset)
        values_np = np.random.randint(0, rows[i], np.sum(offset_np))
        values_numpy.append(values_np)
        value_tensor = tf.convert_to_tensor(values_np)
        value_tensor = hvd.broadcast(value_tensor, root_rank=0)
        value_tensors.append(value_tensor)
        value = tf.placeholder(shape=[None], dtype=tf.int64)
        values.append(value)
        total_indices.append(tf.RaggedTensor.from_row_lengths(value, offset))
    left = batch_size // hvd.size() * rank
    right = batch_size // hvd.size() * (rank + 1)

    weights = []
    for i in range(len(rows)):
        weight = tf.placeholder(shape=(rows[i], cols[i]), dtype=tf.float32)
        weights.append(weight)

    weights_numpy = []
    for i in range(len(rows)):
        weight_numpy = (
            np.ones(rows[i] * cols[i]).reshape([rows[i], cols[i]]).astype(np.float32)
        ) * initial_vals[i]
        weights_numpy.append(weight_numpy)

    # initialize optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    sok_optimizer = sok.OptimizerWrapper(optimizer)

    def step(params):
        embeddings = sok.lookup_sparse(params, total_indices, None, combiners)
        loss = 0
        for i in range(len(embeddings)):
            loss = loss + tf.reduce_sum(embeddings[i])
        grads = tf.gradients(loss, params)
        apply_gradients_op = sok_optimizer.apply_gradients(zip(grads, params))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return apply_gradients_op, loss

    apply_gradients_op, loss = step(sok_vars)

    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op, feed_dict=dict(zip(weights, weights_numpy)))

    broadcast_value_tensors = sess.run(value_tensors)
    for i in range(len(broadcast_value_tensors)):
        values_numpy[i] = broadcast_value_tensors[i]
    indices_records = []
    time_records = []
    for i in range(iters):
        tmp_offset_numpy = []
        tmp_values_numpy = []
        indices_global = []
        for j in range(len(rows)):
            tmp_offset_numpy.append(
                offsets_numpy[j][i * batch_size + left : i * batch_size + right]
            )

            tmp_value_left_offset = np.squeeze(np.sum(offsets_numpy[j][0 : i * batch_size + left]))
            tmp_value_rigth_offset = np.squeeze(
                np.sum(offsets_numpy[j][0 : i * batch_size + right])
            )
            tmp_values_numpy.append(values_numpy[j][tmp_value_left_offset:tmp_value_rigth_offset])

            tmp_global_left_offset = np.squeeze(np.sum(offsets_numpy[j][0 : i * batch_size]))
            tmp_global_right_offset = np.squeeze(np.sum(offsets_numpy[j][0 : (i + 1) * batch_size]))
            indices_global.append(values_numpy[j][tmp_global_left_offset:tmp_global_right_offset])
        indices_records.append(indices_global)
        time.sleep(sleep_seconds)
        sess.run(
            apply_gradients_op,
            feed_dict=dict(zip(offsets + values, tmp_offset_numpy + tmp_values_numpy)),
        )
        time.sleep(sleep_seconds)
        utc_time = datetime.now(pytz.utc)
        time_records.append(utc_time)

        if i > 0:
            time_before = time_records[i - 1]
            dump_keys, dump_values = sok.incremental_model_dump(sok_vars, time_before, sess)

            num_lookups = len(dump_keys)
            indices_before = indices_records[i]
            for lookup_id in range(num_lookups):

                tmp_key = dump_keys[lookup_id]
                indices_np = indices_before[lookup_id]
                indices_np, unique_reverse_indices = np.unique(indices_np, return_index=True)
                indices_np = np.sort(indices_np)
                tmp_key = np.sort(tmp_key)
                np.testing.assert_array_equal(indices_np, tmp_key)
            print("____________iter {} is pass!________________".format(str(i)))

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

from sparse_operation_kit import experiment as sok


if __name__ == "__main__":

    hvd.init()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sok.init()

    rows = [65536 * 10, 65536]
    cols = [128, 4]
    hotness = [10, 3]
    combiners = ["mean", "sum"]
    batch_size = 65536
    iters = 100

    # initial value of embedding table
    weights = []
    for i in range(len(rows)):
        weight_numpy = np.random.rand(rows[i], cols[i]).astype(np.float32)
        weight = tf.convert_to_tensor(weight_numpy)
        # make sure the weight is same on each rank
        weight = hvd.allreduce(weight)
        weights.append(weight)

    # sok variables
    sok_vars = [sok.Variable(w) for w in weights]

    local_indices = []
    local_indices_numpy = []
    for row in rows:
        local_size = row // hvd.size()
        if hvd.rank() < row % hvd.size():
            local_size += 1
        indices_numpy = np.arange(local_size) * hvd.size() + hvd.rank()
        indices = tf.placeholder(shape=[None], dtype=tf.int64)
        local_indices.append(indices)
        local_indices_numpy.append(indices_numpy)

    tf_vars = [tf.Variable(w, use_resource=True) for w in weights]
    # indices

    offsets_numpy = []
    values_numpy = []
    offsets = []
    values = []
    total_indices = []

    for i in range(len(rows)):
        offsets_np = np.random.randint(1, hotness[i] + 1, iters * batch_size)
        offsets_numpy.append(offsets_np)
        values_np = np.random.randint(0, rows[i], np.squeeze(np.sum(offsets_np)))
        values_numpy.append(values_np)

        offset = tf.placeholder(shape=[None], dtype=tf.int64)
        offset = hvd.broadcast(offset, root_rank=0)
        offsets.append(offset)
        value = tf.placeholder(shape=[None], dtype=tf.int64)
        value = hvd.broadcast(value, root_rank=0)
        values.append(value)
        total_indice = tf.RaggedTensor.from_row_lengths(value, offset)
        total_indices.append(total_indice)

    left = batch_size // hvd.size() * hvd.rank()
    right = batch_size // hvd.size() * (hvd.rank() + 1)

    # initialize optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

    def step(params):
        embeddings = sok.lookup_sparse(params, total_indices, combiners)
        loss = 0
        for i in range(len(embeddings)):
            loss = loss + tf.reduce_sum(embeddings[i])
        grads = tf.gradients(loss, params)
        optimizer.apply_gradients(zip(grads, params))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss

    # Do training
    loss1 = []
    ts = []
    t = time.time()

    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    sok_embedding = step(sok_vars)
    for i in range(iters):
        ts.append(time.time() - t)
        t = time.time()
        tmp_offset_numpy = []
        tmp_values_numpy = []
        for j in range(len(rows)):
            tmp_offset_numpy.append(
                offsets_numpy[j][i * batch_size + left : i * batch_size + right]
            )
            tmp_value_left_offset = np.squeeze(np.sum(offsets_numpy[j][0 : i * batch_size + left]))
            tmp_value_rigth_offset = np.squeeze(
                np.sum(offsets_numpy[j][0 : i * batch_size + right])
            )
            tmp_values_numpy.append(values_numpy[j][tmp_value_left_offset:tmp_value_rigth_offset])
        loss = sess.run(
            sok_embedding,
            feed_dict={
                offsets[0]: tmp_offset_numpy[0],
                offsets[1]: tmp_offset_numpy[1],
                values[0]: tmp_values_numpy[0],
                values[1]: tmp_values_numpy[1],
            },
        )
        loss1.append(loss)
        print("-" * 30 + "iteration %d" % i + "-" * 30)
        print("loss:", loss)
    out1 = []
    for i in range(len(sok_vars)):
        out1.append(sok_vars[i].eval(sess))

    def step2(params):
        loss = 0
        embeddings = []
        for i in range(len(params)):
            embedding = tf.nn.embedding_lookup_sparse(
                params[i], total_indices[i].to_sparse(), None, combiner=combiners[i]
            )
            embeddings.append(embedding)
            loss = loss + tf.reduce_sum(embedding)
        grads = tf.gradients(loss, params)
        grads = [hvd.allreduce(grad, op=hvd.Sum) for grad in grads]
        optimizer.apply_gradients(zip(grads, params))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss

    tf_embedding = step2(tf_vars)
    loss2 = []
    for i in range(iters):
        tmp_offset_numpy = []
        tmp_values_numpy = []
        for j in range(len(rows)):
            tmp_offset_numpy.append(
                offsets_numpy[j][i * batch_size + left : i * batch_size + right]
            )
            tmp_value_left_offset = np.squeeze(np.sum(offsets_numpy[j][0 : i * batch_size + left]))
            tmp_value_rigth_offset = np.squeeze(
                np.sum(offsets_numpy[j][0 : i * batch_size + right])
            )
            tmp_values_numpy.append(values_numpy[j][tmp_value_left_offset:tmp_value_rigth_offset])
        loss = sess.run(
            tf_embedding,
            feed_dict={
                offsets[0]: tmp_offset_numpy[0],
                offsets[1]: tmp_offset_numpy[1],
                values[0]: tmp_values_numpy[0],
                values[1]: tmp_values_numpy[1],
            },
        )
        loss2.append(loss)

        print("-" * 30 + "iteration %d" % i + "-" * 30)
        print("tf loss:", loss)
    out2 = []
    for i, v in enumerate(tf_vars):
        tmp_out = sess.run(
            tf.nn.embedding_lookup(v, local_indices[i]),
            feed_dict={local_indices[i]: local_indices_numpy[i]},
        )
        out2.append(tmp_out)

    # Check results
    diff = 0
    for i in range(len(out1)):
        length = out1[i] ** 2 + out2[i] ** 2 + 1e-8
        diff = diff + np.sum((out1[i] - out2[i]) ** 2 / length)
    print("[SOK INFO] diff:", diff)
    assert diff < 1e-6

    diff = 0
    for i in range(iters):
        # normalize
        length = loss1[i] ** 2 + loss2[i] ** 2 + 1e-8
        diff = diff + (loss1[i] - loss2[i]) ** 2 / length
    print("[SOK INFO] loss diff:", diff, "hvd.rank() = ", hvd.rank())
    assert diff < 1e-6

    print("[SOK INFO] lookup_sparse distributed test passed")
    ts = ts[5:]
    print("[SOK INFO] Average time: %f ms/iteration" % (sum(ts) / len(ts) * 1000))

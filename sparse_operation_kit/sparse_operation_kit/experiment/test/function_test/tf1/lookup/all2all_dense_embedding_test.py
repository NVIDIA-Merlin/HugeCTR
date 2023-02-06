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
    # initialize
    hvd.init()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sok.init()

    row = 65536 * 10
    col = 128
    batch_size = 65536
    iters = 100

    # initial value of embedding table
    weight_numpy = np.random.rand(row, col).astype(np.float32)
    weight = tf.convert_to_tensor(weight_numpy)
    # make sure the weight is same on each rank
    weight = hvd.allreduce(weight)

    # sok variable
    sok_var = sok.Variable(weight)
    tf_var = tf.Variable(weight, use_resource=True)
    local_size = row // hvd.size()
    if hvd.rank() < row % hvd.size():
        local_size += 1
    local_indices_numpy = np.arange(local_size) * hvd.size() + hvd.rank()
    local_indices = tf.placeholder(shape=[None], dtype=tf.int64)

    # indices
    total_indices_numpy = np.random.randint(0, row, [iters, batch_size])
    total_indices = tf.placeholder(shape=[None], dtype=tf.int64)
    # make sure the total_indices is same on each rank
    total_indices = hvd.broadcast(total_indices, root_rank=0)
    left = batch_size // hvd.size() * hvd.rank()
    right = batch_size // hvd.size() * (hvd.rank() + 1)

    # initialize optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)

    # graph with sok.all2all_dense_embedding
    def step(param, indices, optimizer):
        embedding = sok.all2all_dense_embedding(param, indices)
        loss = tf.reduce_sum(embedding)
        grads = tf.gradients(loss, [param])
        optimizer.apply_gradients(zip(grads, [param]))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss

    sok_embedding = step(sok_var, total_indices, optimizer)
    # Do training with sok.Variable
    loss1 = []
    ts = []
    t = time.time()
    for i in range(iters):
        ts.append(time.time() - t)
        t = time.time()
        loss = sess.run(
            sok_embedding,
            feed_dict={total_indices: total_indices_numpy[i, left:right]},
        )
        loss1.append(loss)
        print("-" * 30 + "iteration %d" % i + "-" * 30)
        print("hvd.rank() ", hvd.rank(), " loss:", loss)
    out1 = sok_var.eval(sess)

    # graph with tf.nn.embedding_lookup
    def step2(param, indices, optimizer):
        embedding = tf.nn.embedding_lookup(param, indices)
        loss = tf.reduce_sum(embedding)
        grads = tf.gradients(loss, [param])
        grads = [hvd.allreduce(grad, op=hvd.Sum) for grad in grads]
        optimizer.apply_gradients(zip(grads, [param]))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss

    # Do training with tf.Variable
    tf_embedding = (step2(tf_var, total_indices, optimizer),)
    loss2 = []
    for i in range(iters):
        loss = sess.run(
            tf_embedding,
            feed_dict={total_indices: total_indices_numpy[i, left:right]},
        )
        loss2.append(loss[0])
        print("-" * 30 + "iteration %d" % i + "-" * 30)
        print("hvd.rank() = ", hvd.rank(), " tf loss:", loss)
    out2 = sess.run(
        tf.nn.embedding_lookup(tf_var, local_indices),
        feed_dict={tf_var: weight_numpy, local_indices: local_indices_numpy},
    )
    # Check results
    length = out1**2 + out2**2 + 1e-8
    diff = np.sum((out1 - out2) ** 2 / length)
    print("[SOK INFO] diff:", diff)
    assert diff < 1e-6

    diff = 0
    for i in range(iters):
        length = loss1[i] ** 2 + loss2[i] ** 2 + 1e-8
        diff = diff + (loss1[i] - loss2[i]) ** 2 / length
    print("[SOK INFO] loss diff:", diff)
    assert diff < 1e-6

    print("[SOK INFO] all2all_dense_embedding test passed")
    ts = ts[5:]
    print("[SOK INFO] Average time: %f ms/iteration" % (sum(ts) / len(ts) * 1000))

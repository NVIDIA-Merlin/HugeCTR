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
import argparse
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from sparse_operation_kit import experiment as sok


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--hotness", type=int, default=2)
    parser.add_argument("--combiner", type=str, default="sum")
    parser.add_argument("--key_space", type=int, default=1024 * 1024)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    args.iters = max(args.iters, 10)

    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    sok.init()

    offsets = np.random.randint(1, args.hotness + 1, args.iters * args.batch_size)
    offsets = tf.convert_to_tensor(offsets, dtype=tf.int64)
    values = np.random.randint(0, args.key_space, tf.reduce_sum(offsets))
    values = tf.convert_to_tensor(values, dtype=tf.int64)
    total_indices = tf.RaggedTensor.from_row_lengths(values, offsets)

    v = tf.Variable(tf.random.normal(shape=[args.key_space, args.dim]), dtype=tf.float32)
    sok_v = sok.Variable(v)

    @tf.function
    def sok_step(param, indices):
        with tf.GradientTape() as tape:
            embedding = sok.lookup_sparse(param, indices, args.hotness, args.combiner)
            loss = tf.reduce_sum(embedding)
        grads = tape.gradient(loss, [param])
        return loss, grads

    ts = []
    t = time.time()
    for i in range(args.iters):
        ts.append(time.time() - t)
        t = time.time()
        left = args.batch_size // hvd.size() * hvd.rank()
        left += i * args.batch_size
        right = args.batch_size // hvd.size() * (hvd.rank() + 1)
        right += i * args.batch_size
        loss, _ = sok_step(sok_v, total_indices[left:right])
        loss = loss.numpy()
    sok_result = sum(ts[5:]) / (args.iters - 5) * 1000

    @tf.function
    def tf_step(param, indices):
        with tf.GradientTape() as tape:
            embedding = tf.nn.embedding_lookup_sparse(param, indices, None, combiner=args.combiner)
            loss = tf.reduce_sum(embedding)
        grads = tape.gradient(loss, [param])
        return loss, grads

    ts = []
    t = time.time()
    for i in range(args.iters):
        ts.append(time.time() - t)
        t = time.time()
        left = args.batch_size // hvd.size() * hvd.rank()
        left += i * args.batch_size
        right = args.batch_size // hvd.size() * (hvd.rank() + 1)
        right += i * args.batch_size
        sp_ids = total_indices[left:right].to_sparse()
        loss, _ = tf_step(v, sp_ids)
        loss = loss.numpy()
    tf_result = sum(ts[5:]) / (args.iters - 5) * 1000

    print("---------------------------------------------")
    print("* batch_size          : %d" % args.batch_size)
    print("* local batch_size    : %d" % (args.batch_size // hvd.size()))
    print("* hotness             : %d" % args.hotness)
    print("* combiner            : %s" % args.combiner)
    print("* key_space           : %d" % args.key_space)
    print("* dim                 : %d" % args.dim)
    print("---------------------------------------------")
    print("* sok.lookup_sparse   : %.3f ms/iter" % sok_result)
    print("* tf lookup_sparse    : %.3f ms/iter" % tf_result)
    print("---------------------------------------------")

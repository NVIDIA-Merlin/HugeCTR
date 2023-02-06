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
    parser.add_argument("--key_space", type=int, default=1024 * 1024)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    args.iters = max(args.iters, 10)

    physical_devices = tf.config.list_physical_devices("GPU")
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    total_indices = np.random.randint(0, args.key_space, [args.iters, args.batch_size])
    total_indices = tf.convert_to_tensor(total_indices, dtype=tf.int64)

    @tf.function
    def sok_step(v, indices):
        with tf.GradientTape() as tape:
            embedding = tf.nn.embedding_lookup(v, indices)
            loss = tf.reduce_sum(embedding)
        grads = tape.gradient(loss, [v])
        return loss, grads

    v = sok.DynamicVariable(dimension=args.dim, initializer="13")
    ts = []
    t = time.time()
    for i in range(args.iters):
        ts.append(time.time() - t)
        t = time.time()
        loss, _ = sok_step(v, total_indices[i])
        loss = loss.numpy()
    sok_result = sum(ts[5:]) / (args.iters - 5) * 1000

    def tf_step(v, indices):
        with tf.GradientTape() as tape:
            embedding = tf.nn.embedding_lookup(v, indices)
            loss = tf.reduce_sum(embedding)
        grads = tape.gradient(loss, [v])
        return loss, grads

    v = tf.Variable(tf.constant(13, shape=[args.key_space, args.dim], dtype=tf.float32))
    ts = []
    t = time.time()
    for i in range(args.iters):
        ts.append(time.time() - t)
        t = time.time()
        loss, _ = tf_step(v, total_indices[i])
        loss = loss.numpy()
    tf_result = sum(ts[5:]) / (args.iters - 5) * 1000

    print("---------------------------------------------")
    print("* batch_size          : %d" % args.batch_size)
    print("* key_space           : %d" % args.key_space)
    print("* dim                 : %d" % args.dim)
    print("---------------------------------------------")
    print("* sok.DynamicVariable : %.3f ms/iter" % sok_result)
    print("* tf.Variable         : %.3f ms/iter" % tf_result)
    print("---------------------------------------------")

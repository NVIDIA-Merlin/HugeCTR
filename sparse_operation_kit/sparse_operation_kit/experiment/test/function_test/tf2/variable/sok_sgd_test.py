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

import numpy as np
import tensorflow as tf

from sparse_operation_kit import experiment as sok


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices("GPU")
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    dim = 128
    vocab_size = 1024 * 128
    batch = 8192

    sok_var = sok.DynamicVariable(dimension=dim)
    sok_optimizer = sok.SGD(lr=1.0)

    indices_val = [idx for idx in range(vocab_size)]
    table_val = tf.nn.embedding_lookup(sok_var, indices_val)
    tf_var = tf.Variable(table_val)
    tf_optimizer = tf.optimizers.SGD(learning_rate=1.0)

    @tf.function
    def sok_step(indices, weight, var):
        with tf.GradientTape() as tape:
            emb = tf.nn.embedding_lookup(var, indices)
            emb_mul = emb * weight
            loss = tf.reduce_sum(emb_mul)
        grads = tape.gradient(loss, [var])
        sok_optimizer.apply_gradients(zip(grads, [var]))
        return loss

    @tf.function
    def tf_step(indices, weight, var):
        with tf.GradientTape() as tape:
            emb = tf.nn.embedding_lookup(var, indices)
            emb_mul = emb * weight
            loss = tf.reduce_sum(emb_mul)
        grads = tape.gradient(loss, [var])
        tf_optimizer.apply_gradients(zip(grads, [var]))
        return loss

    num = np.random.randint(1, batch + 1, 1)[0]
    for i in range(100):
        print("---------------------Iter %d---------------------" % i)
        indices_val = np.random.randint(0, vocab_size, num).astype(np.int64)
        indices_val = tf.convert_to_tensor(indices_val, dtype=tf.int64)
        weight_val = np.random.rand(num, dim).astype(np.float32)
        weight_val = tf.convert_to_tensor(weight_val, dtype=tf.float32)
        sok_loss = sok_step(indices_val, weight_val, sok_var)
        tf_loss = tf_step(indices_val, weight_val, tf_var)
        print(sok_loss, tf_loss)

    indices_val = [idx for idx in range(vocab_size)]
    table_val = tf.nn.embedding_lookup(sok_var, indices_val)
    diff = tf.reduce_mean((table_val - tf_var) ** 2.0)
    assert diff < 1e-8
    print("[SOK INFO] Test variable with sok.SGD successfully")

    # ----------------------------Test eager mode----------------------------

    def sok_step_eager(indices, weight, var):
        with tf.GradientTape() as tape:
            emb = tf.nn.embedding_lookup(var, indices)
            emb_mul = emb * weight
            loss = tf.reduce_sum(emb_mul)
        grads = tape.gradient(loss, [var])
        sok_optimizer.apply_gradients(zip(grads, [var]))
        return loss

    def tf_step_eager(indices, weight, var):
        with tf.GradientTape() as tape:
            emb = tf.nn.embedding_lookup(var, indices)
            emb_mul = emb * weight
            loss = tf.reduce_sum(emb_mul)
        grads = tape.gradient(loss, [var])
        tf_optimizer.apply_gradients(zip(grads, [var]))
        return loss

    for i in range(100):
        num = np.random.randint(1, batch + 1, 1)[0]
        print("---------------------Iter %d---------------------" % i)
        indices_val = np.random.randint(0, vocab_size, num).astype(np.int64)
        indices_val = tf.convert_to_tensor(indices_val, dtype=tf.int64)
        weight_val = np.random.rand(num, dim).astype(np.float32)
        weight_val = tf.convert_to_tensor(weight_val, dtype=tf.float32)
        sok_loss = sok_step_eager(indices_val, weight_val, sok_var)
        tf_loss = tf_step_eager(indices_val, weight_val, tf_var)
        print(sok_loss, tf_loss)

    indices_val = [idx for idx in range(vocab_size)]
    table_val = tf.nn.embedding_lookup(sok_var, indices_val)
    diff = tf.reduce_mean((table_val - tf_var) ** 2.0)
    assert diff < 1e-8
    print("[SOK INFO] Test variable with sok.SGD and eager mode successfully")

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
import horovod.tensorflow as hvd

from sparse_operation_kit import experiment as sok


if __name__ == "__main__":

    v = sok.DynamicVariable(dimension=3, initializer="13")
    print("v.shape:", v.shape)
    print("v.size:", v.size)

    indices = tf.convert_to_tensor([0, 1, 2**40], dtype=tf.int64)

    with tf.GradientTape() as tape:
        embedding = tf.nn.embedding_lookup(v, indices)
        print("embedding:\n", embedding)
        loss = tf.reduce_sum(embedding)

    grads = tape.gradient(loss, [v])
    print("grad:\n", grads[0])

    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    optimizer = sok.OptimizerWrapper(optimizer)
    optimizer.apply_gradients(zip(grads, [v]))

    embedding = tf.nn.embedding_lookup(v, indices)
    print("embedding:\n", embedding)

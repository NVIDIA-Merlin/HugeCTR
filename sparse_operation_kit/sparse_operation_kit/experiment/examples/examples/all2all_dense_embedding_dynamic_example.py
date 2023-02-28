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
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")  # nopep8

    sok.init()

    # size of embedding table
    row = 32
    # dimension of embedding table
    col = 4

    # The distributed version
    param = sok.DynamicVariable(dimension=col, initializer="13", dtype=tf.float32)  # nopep8
    print("param:\n", param)

    total_indices = np.arange(row)
    total_indices = tf.convert_to_tensor(total_indices, dtype=tf.int64)

    # Divide indices into n parts
    left = row // hvd.size() * hvd.rank()
    right = row // hvd.size() * (hvd.rank() + 1)
    indices = total_indices[left:right]
    print("indice:\n", indices)

    with tf.GradientTape() as tape:
        embedding = sok.all2all_dense_embedding(param, indices)
        print("embedding:\n", embedding)
        loss = tf.reduce_sum(embedding)
    grads = tape.gradient(loss, [param])
    print("grads:\n", grads[0])

    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    optimizer = sok.OptimizerWrapper(optimizer)
    optimizer.apply_gradients(zip(grads, [param]))
    print("param:\n", param)
    embedding = sok.all2all_dense_embedding(param, indices)
    print("embedding:\n", embedding)
    print("param.size:\n", param.size)

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

    # Localized mode
    # If there are 2 GPUs in total, the shape of v1 on GPU0 will be [17, 3] and the shape
    # on GPU1 will be [0, 3]
    v1 = sok.Variable(
        np.arange(17 * 3).reshape(17, 3), dtype=tf.float32, mode="localized:0"
    )  # nopep8
    v2 = sok.Variable(
        np.arange(7 * 5).reshape(7, 5), dtype=tf.float32, mode="localized:%d" % (hvd.size() - 1)
    )  # nopep8
    print("v1:\n", v1)
    print("v2:\n", v2)

    indices1 = tf.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], values=[1, 1, 3, 4, 5], dense_shape=[2, 3]
    )
    print("indices1:\n", indices1)
    # indices1: batch_size=2, max_hotness=3
    # [[1, 1]
    #  [3, 4, 5]]

    indices2 = tf.SparseTensor(
        indices=[[0, 0], [1, 0], [1, 1]], values=[1, 2, 3], dense_shape=[2, 2]
    )
    print("indices2:\n", indices2)
    # indices2: batch_size=2, max_hotness=2
    # [[1]
    #  [2, 3]]

    with tf.GradientTape() as tape:
        embeddings = sok.lookup_sparse(
            [v1, v2], [indices1, indices2], None, combiners=["sum", "sum"]
        )
        loss = 0.0
        for i, embedding in enumerate(embeddings):
            loss += tf.reduce_sum(embedding)
            print("embedding%d:\n" % (i + 1), embedding)
        # embedding1: [[6,  8,  10]
        #              [36, 39, 42]]
        # embedding2: [[5,  6,  7,  8,  9
        #              [25, 27, 29, 31, 33]]

    # If there are 2 GPUs in total
    # GPU0:
    #   In Distributed mode: shape of grad of v1 will be [1, 3], shape of grad of v2 will be [1, 5]
    #   In Localized mode: shape of grad of v1 will be [4, 3], grad of v2 will None
    # GPU1:
    #   In Distributed mode: shape of grad of v1 will be [3, 3], shape of grad of v2 will be [2, 5]
    #   In Localized mode: grad of v1 will be None, shape of grad of v2 will be [3, 5]
    grads = tape.gradient(loss, [v1, v2])
    for i, grad in enumerate(grads):
        print("grad%d:\n" % (i + 1), grad)

    # Use tf.keras.optimizer to optimize the sok.Variable
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    optimizer.apply_gradients(zip(grads, [v1, v2]))
    print("v1:\n", v1)
    print("v2:\n", v2)

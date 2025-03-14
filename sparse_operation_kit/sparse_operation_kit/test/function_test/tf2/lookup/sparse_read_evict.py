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


def check_overlap(arr_list, arr2):
    for i in range(len(arr_list)):
        # Use np.in1d to check if any element of arr2 is in arr_list[i]
        overlap = np.in1d(arr2, arr_list[i])
        tmp_overlap = np.any(overlap)
        if tmp_overlap:
            return tmp_overlap
    return False


if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    sok.init()
    iter_num = 5
    input_length = 8192

    evict_keys_list = []
    evict_values_list = []
    # sok variables
    sok_var = sok.DynamicVariable(
        dimension=16,
        var_type="hybrid",
        initializer=str(11),
        init_capacity=input_length,
        max_capacity=input_length * 2,
    )

    optimizer = tf.optimizers.SGD(learning_rate=1.0, momentum=0.9)

    if sok.tf_version[0] == 2 and sok.tf_version[1] >= 17:
        import tf_keras as tfk

        optimizer_for_sok = tfk.optimizers.legacy.SGD(learning_rate=1.0)
    else:
        optimizer_for_sok = tf.optimizers.SGD(learning_rate=1.0)
    sok_optimizer = sok.OptimizerWrapper(optimizer_for_sok)

    for i in range(iter_num):
        indices_values = tf.constant(
            range(i * input_length, (i + 1) * input_length), dtype=tf.int64
        )
        indices = tf.ragged.constant(indices_values, dtype=tf.int64)

        with tf.GradientTape() as tape:
            embedding_first, evict_key, evict_value = sok.sparse_read_and_evict(sok_var, indices)
            loss = tf.reduce_sum(embedding_first)
        grads = tape.gradient(loss, [sok_var])
        grad_pair = zip(grads, [sok_var])
        sok_optimizer.apply_gradients(grad_pair)

        evict_key_np = evict_key.numpy()
        evict_value_np = evict_value.numpy()

        if i > 0:
            assert not check_overlap(
                evict_keys_list, evict_key_np
            ), "Not all indices are within the specified range."
        assert np.all(evict_value_np == 10), "Not all values are updated correctly."
        evict_keys_list.append(evict_key_np)
        evict_values_list.append(evict_value_np)

    print("[SOK INFO] : sparse_read_evict run success!")

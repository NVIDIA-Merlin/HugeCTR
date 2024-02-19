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
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
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
    first_values = np.arange(16, dtype=np.int64)
    second_values = np.arange(16, 24, dtype=np.int64)

    indices = tf.placeholder(shape=[None], dtype=tf.int64)

    # initialize optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    sok_optimizer = sok.OptimizerWrapper(optimizer)

    embedding, evict_key, evict_value = sok.sparse_read_and_evict(sok_var, indices)
    loss = tf.reduce_sum(embedding)
    grads = tf.gradients(loss, [sok_var])
    apply_gradients_op = sok_optimizer.apply_gradients(zip(grads, [sok_var]))

    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    for i in range(iter_num):
        indices_values = np.arange(i * input_length, (i + 1) * input_length, dtype=np.int64)
        embedding_np, evict_key_np, evict_value_np, _ = sess.run(
            [embedding, evict_key, evict_value, apply_gradients_op],
            feed_dict=dict(zip([indices], [indices_values])),
        )

        if i > 0:
            assert not check_overlap(
                evict_keys_list, evict_key_np
            ), "Not all indices are within the specified range."
        assert np.all(evict_value_np == 10), "Not all values are updated correctly."
        evict_keys_list.append(evict_key_np)
        evict_values_list.append(evict_value_np)

    print("[SOK INFO] : sparse_read_evict run success!")

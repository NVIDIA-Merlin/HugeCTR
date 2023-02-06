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

    vocab_size = 1023 * 1024 * 2
    dim = 128

    with tf.device("CPU"):
        indices = tf.convert_to_tensor([i for i in range(vocab_size)], dtype=tf.int64)
        values = tf.convert_to_tensor(np.random.rand(vocab_size, dim), dtype=tf.float32)
    v = sok.DynamicVariable(dimension=dim)

    # Test assign
    sok.assign(v, indices, values)
    shape = v.size
    assert shape[0] == vocab_size

    # Test export
    ex_indices, ex_values = sok.export(v)
    with tf.device("CPU"):
        out = tf.gather(values, ex_indices)
        diff = tf.reduce_sum((ex_values - out) ** 2.0)
        assert diff < 1e-6

    print("[SOK INFO] Test assign and export successfully.")

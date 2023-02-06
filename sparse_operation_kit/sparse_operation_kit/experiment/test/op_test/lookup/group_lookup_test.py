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


def test():
    v1 = tf.Variable(np.arange(12).reshape(3, 4), dtype=tf.float32)
    v2 = tf.Variable(np.arange(15).reshape(5, 3), dtype=tf.float32)
    indices1 = tf.convert_to_tensor([0, 1], dtype=tf.int32)
    indices2 = tf.convert_to_tensor([1, 2, 3], dtype=tf.int32)
    outputs = sok.raw_ops.group_lookup([v1.handle, v2.handle], [indices1, indices2])

    assert len(outputs) == 2

    output = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    err = tf.reduce_mean((outputs[0] - output) ** 2)
    assert err < 1e-8

    output = [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
    err = tf.reduce_mean((outputs[1] - output) ** 2)
    assert err < 1e-8


if __name__ == "__main__":
    op_name = "group_lookup"
    if not hasattr(sok.raw_ops, op_name):
        raise RuntimeError("There is no op called " + op_name)

    test()

    print("[SOK INFO] Test of %s passed." % (op_name))

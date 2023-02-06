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
    embedding = tf.convert_to_tensor(np.arange(3 * 4).reshape(3, 4), dtype=tf.float32)
    order = tf.convert_to_tensor([1, 2, 0], dtype=tf.int32)
    output = sok.raw_ops.reorder(embedding, order)

    gt = [[8.0, 9.0, 10.0, 11.0], [0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    err = tf.reduce_mean((gt - output) ** 2)
    assert err < 1e-8


if __name__ == "__main__":
    op_name = "reorder"
    if not hasattr(sok.raw_ops, op_name):
        raise RuntimeError("There is no op called " + op_name)

    test()

    print("[SOK INFO] Test of %s passed." % (op_name))

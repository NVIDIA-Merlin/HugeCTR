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
    handle = sok.raw_ops.dummy_var_handle(
        shared_name="Var_0", shape=[None, 128], key_type=tf.int64, dtype=tf.float32
    )
    sok.raw_ops.dummy_var_initialize(
        handle,
        initializer=[2.71828],
        var_type="",
        unique_name="",
        key_type=tf.int64,
        dtype=tf.float32,
    )
    indices = tf.convert_to_tensor([0, 1, 2**40], dtype=tf.int64)
    shape = sok.raw_ops.dummy_var_shape(handle)
    assert shape[0] == 0
    assert shape[1] == 128
    embedding_vector = sok.raw_ops.dummy_var_sparse_read(handle, indices)
    shape = sok.raw_ops.dummy_var_shape(handle)
    assert shape[0] == 3
    assert shape[1] == 128


if __name__ == "__main__":
    op_name = "dummy_var_shape"
    if not hasattr(sok.raw_ops, op_name):
        raise RuntimeError("There is no op called " + op_name)

    test()

    print("[SOK INFO] Test of %s passed." % (op_name))

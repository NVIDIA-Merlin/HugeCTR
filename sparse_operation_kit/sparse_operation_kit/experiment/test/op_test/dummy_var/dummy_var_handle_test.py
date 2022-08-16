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
    handle1 = sok.raw_ops.dummy_var_handle(
        container="DummyVarTestContainer",
        shared_name="TestVar_0",
        shape=[None, 2],
        key_type=tf.int32,
        dtype=tf.float32,
    )
    assert handle1.dtype == tf.resource

    @tf.function
    def step():
        return sok.raw_ops.dummy_var_handle(
            container="DummyVarTestContainer",
            shared_name="TestVar_1",
            shape=[None, 128],
            key_type=tf.int64,
            dtype=tf.float32,
        )

    handle2 = step()
    assert handle2.dtype == tf.resource


if __name__ == "__main__":

    op_name = "dummy_var_handle"
    if not hasattr(sok.raw_ops, op_name):
        raise RuntimeError("There is no op called " + op_name)

    test()

    print("[SOK INFO] Test of %s passed." % (op_name))

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
    grad1 = tf.convert_to_tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=tf.float32
    )
    row_length1 = tf.convert_to_tensor([1, 2], dtype=tf.int64)
    grad2 = tf.convert_to_tensor([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]], dtype=tf.float32)
    row_length2 = tf.convert_to_tensor([1, 1], dtype=tf.int64)
    shape = tf.convert_to_tensor([16, 16], dtype=tf.int64)
    outputs = sok.raw_ops.postprocessing_backward(
        emb_vec_grad=[grad1, grad2],
        emb_vec_buffer_shape=shape,
        row_lengths=[row_length1, row_length2],
        combiners=["sum", "sum"],
        hotness=[2, 3],
        shard=[-1, -1],
        dimensions=[5, 3],
        rank=0,
        num_ranks=2,
        id_in_local_rank=0,
        num_gpus=2,
        Tindices=tf.int64,
        # Toffsets=tf.int64,
    )

    g = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    for i in range(len(g)):
        outputs[0][i] == g[i]
        outputs[1][i] == g[i]


if __name__ == "__main__":

    op_name = "postprocessing_backward"
    if not hasattr(sok.raw_ops, op_name):
        raise RuntimeError("There is no op called " + op_name)

    test()

    print("[SOK INFO] Test of %s passed." % (op_name))

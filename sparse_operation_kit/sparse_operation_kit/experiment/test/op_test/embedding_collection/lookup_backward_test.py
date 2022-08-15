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
    emb_vec_buffer_grad = tf.convert_to_tensor(
        [0.0, 1.0, 2.0, 9.0, 11.0, 13.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        dtype=tf.float32,
    )
    model_key = tf.convert_to_tensor([0, 1, 2, 1, 2], dtype=tf.int64)
    model_offsets = tf.convert_to_tensor([0, 1, 3, 4, 5], dtype=tf.uint32)
    indices, grads = sok.raw_ops.lookup_backward(
        [emb_vec_buffer_grad],
        model_key,
        model_offsets,
        num_lookups=2,
        combiners=["sum", "sum"],
        hotness=[2, 2],
        shard=[0, 0],
        dimensions=[3, 5],
        rank=0,
        num_ranks=1,
        id_in_local_rank=0,
        Toffsets=tf.int64,
    )
    assert len(indices) == 2
    for i, item in enumerate([0, 1, 2]):
        assert indices[0][i] == item
    for i, item in enumerate([1, 2]):
        assert indices[1][i] == item

    assert len(grads) == 2
    assert grads[0].shape[0] == 3
    assert grads[0].shape[1] == 3
    for i, item in enumerate([0.0, 1.0, 2.0, 9.0, 11.0, 13.0, 9.0, 11.0, 13.0]):
        assert grads[0][i // 3, i % 3] == item

    assert grads[1].shape[0] == 2
    assert grads[1].shape[1] == 5
    for i, item in enumerate([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]):
        assert grads[1][i // 5, i % 5] == item


if __name__ == "__main__":

    op_name = "lookup_backward"
    if not hasattr(sok.raw_ops, op_name):
        raise RuntimeError("There is no op called " + op_name)

    test()

    print("[SOK INFO] Test of %s passed." % (op_name))

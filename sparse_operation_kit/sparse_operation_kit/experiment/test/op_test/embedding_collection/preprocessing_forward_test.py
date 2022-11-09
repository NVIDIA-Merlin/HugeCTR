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

import tensorflow as tf
from sparse_operation_kit import experiment as sok


def test():
    key1 = tf.convert_to_tensor([0, 1, 2], dtype=tf.int32)
    key2 = tf.convert_to_tensor([0, 1], dtype=tf.int32)
    offset1 = tf.convert_to_tensor([1, 2], dtype=tf.int32)
    offset2 = tf.convert_to_tensor([1, 1], dtype=tf.int32)
    key_send_buffer, offset_send_buffer = sok.raw_ops.preprocessing_forward(
        [key1, key2],
        [offset1, offset2],
        combiners=["sum", "sum"],
        shard=[-1, -1],
        dimensions=[3, 5],
        rank=0,
        num_ranks=1,
        id_in_local_rank=0,
        num_gpus=1,
    )

    key = tf.concat([key1, key2], 0)
    assert len(key_send_buffer.shape) == 1
    assert key_send_buffer.shape[0] == key.shape[0]
    for i in range(len(key)):
        assert key[i] == key_send_buffer[i]

    offset = tf.concat([offset1, offset2], 0)
    assert len(offset_send_buffer.shape) == 1
    assert offset_send_buffer.shape[0] == offset.shape[0]
    for i in range(len(offset)):
        assert offset[i] == offset_send_buffer[i]


if __name__ == "__main__":

    op_name = "preprocessing_forward"
    if not hasattr(sok.raw_ops, op_name):
        raise RuntimeError("There is no op called " + op_name)

    test()

    print("[SOK INFO] Test of %s passed." % (op_name))

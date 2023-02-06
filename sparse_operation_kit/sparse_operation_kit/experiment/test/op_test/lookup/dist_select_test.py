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
    indices = tf.convert_to_tensor(np.arange(19), dtype=tf.int32)
    output, order, splits = sok.raw_ops.dist_select(indices, num_splits=8)

    assert len(output) == 19
    for i, item in enumerate([0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]):
        assert output[i] == item

    assert len(order) == 19
    for i, item in enumerate([0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]):
        assert order[i] == item

    assert len(splits) == 8
    for i, item in enumerate([3, 3, 3, 2, 2, 2, 2, 2]):
        assert splits[i] == item


if __name__ == "__main__":
    op_name = "dist_select"
    if not hasattr(sok.raw_ops, op_name):
        raise RuntimeError("There is no op called " + op_name)

    test()

    print("[SOK INFO] Test of %s passed." % (op_name))

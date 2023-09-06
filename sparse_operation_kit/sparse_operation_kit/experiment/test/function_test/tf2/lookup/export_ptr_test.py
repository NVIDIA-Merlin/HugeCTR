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

from sparse_operation_kit import experiment as sok


if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    sok.init()

    rows = [8192 * 10, 8192]
    cols = [128, 4]
    hotness = [10, 3]
    combiners = ["sum", "sum"]
    batch_size = 8192
    iters = 100
    initial_vals = [13, 17]

    # sok variables
    sok_vars = [
        sok.DynamicVariable(
            dimension=cols[i],
            var_type="hybrid",
            initializer=str(initial_vals[i]),
            init_capacity=1024 * 1024,
            max_capacity=1024 * 1024,
        )
        for i in range(len(cols))
    ]
    print("HKV var created")
    for i,sok_var in enumerate(sok_vars):
        print(" hvd.rank() = ", hvd.rank(),"i = ",i," sok_var table value =  ",sok_var.dynamic_table_ptr)



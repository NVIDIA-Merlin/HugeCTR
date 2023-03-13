"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
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

import hierarchical_parameter_server as hps
import tensorflow as tf
import os
import numpy as np
import struct
import json
import pytest
import time

NUM_GPUS = 1
VOCAB_SIZE = 10000
EMB_VEC_SIZE = 16
NUM_QUERY_KEY = 26
EMB_VEC_DTYPE = np.float32
TF_KEY_TYPE = tf.int32
MAX_BATCH_SIZE = 256
NUM_ITERS = 100
NUM_TABLES = 100
USE_CONTEXT_STREAM = True

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(NUM_GPUS)))

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.threading.set_inter_op_parallelism_threads(1)


def test_hps_table_fusion():
    os.system("python3 create_model_for_table_fusion.py")
    model = tf.keras.models.load_model(str(NUM_TABLES) + "_table.savedmodel")
    inputs_seq = []
    for _ in range(NUM_ITERS + 1):
        inputs = []
        for i in range(NUM_TABLES):
            inputs.append(
                np.random.randint(
                    i * VOCAB_SIZE, (i + 1) * VOCAB_SIZE, (MAX_BATCH_SIZE, NUM_QUERY_KEY)
                ).astype(np.int32)
            )
        inputs_seq.append(inputs)
    preds = model(inputs_seq[0])
    start = time.time()
    for i in range(NUM_ITERS):
        print("-" * 20, "Step {}".format(i), "-" * 20)
        preds = model(inputs_seq[i + 1])
    end = time.time()
    print(
        "[INFO] Elapsed time for "
        + str(NUM_ITERS)
        + " iterations: "
        + str(end - start)
        + " seconds"
    )

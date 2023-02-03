# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
EMB_VEC_SIZE = 128
NUM_QUERY_KEY = 26
EMB_VEC_DTYPE = np.float32
TF_KEY_TYPE = tf.int64
MAX_BATCH_SIZE = 16
NUM_ITERS = 1000
NUM_TABLES = 8
USE_CONTEXT_STREAM = True

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(NUM_GPUS)))
hps_config = {
    "supportlonglong": True,
    "models": [
        {
            "model": "context_stream",
            "sparse_files": [],
            "num_of_worker_buffer_in_pool": 3,
            "embedding_table_names": [],
            "embedding_vecsize_per_table": [],
            "maxnum_catfeature_query_per_table_per_sample": [],
            "default_value_for_each_table": [0.0],
            "deployed_device_list": [0],
            "max_batch_size": MAX_BATCH_SIZE,
            "cache_refresh_percentage_per_iteration": 0.0,
            "hit_rate_threshold": 1.0,
            "gpucacheper": 1.0,
            "gpucache": True,
            "use_static_table": True,
            "use_context_stream": True,
        }
    ],
}


def generate_embedding_tables(hugectr_sparse_model, vocab_size, embedding_vec_size):
    os.system("mkdir -p {}".format(hugectr_sparse_model))
    with open("{}/key".format(hugectr_sparse_model), "wb") as key_file, open(
        "{}/emb_vector".format(hugectr_sparse_model), "wb"
    ) as vec_file:
        for key in range(vocab_size):
            vec = 0.00025 * np.ones((embedding_vec_size,)).astype(np.float32)
            key_struct = struct.pack("q", key)
            vec_struct = struct.pack(str(embedding_vec_size) + "f", *vec)
            key_file.write(key_struct)
            vec_file.write(vec_struct)


def set_up_model_files(num_tables, use_context_stream):
    if use_context_stream:
        hps_config["models"][0]["use_context_stream"] = True
    for i in range(num_tables):
        table_name = "table" + str(i)
        model_file_name = "embeddings/" + table_name
        generate_embedding_tables(model_file_name, VOCAB_SIZE, EMB_VEC_SIZE)
        hps_config["models"][0]["sparse_files"].append(model_file_name)
        hps_config["models"][0]["embedding_table_names"].append(table_name)
        hps_config["models"][0]["embedding_vecsize_per_table"].append(EMB_VEC_SIZE)
        hps_config["models"][0]["maxnum_catfeature_query_per_table_per_sample"].append(
            NUM_QUERY_KEY
        )

    hps_config_json_object = json.dumps(hps_config, indent=4)
    with open("hps_for_context_stream.json", "w") as outfile:
        outfile.write(hps_config_json_object)


class InferenceModel(tf.keras.models.Model):
    def __init__(self, num_tables, **kwargs):
        super(InferenceModel, self).__init__(**kwargs)
        self.lookup_layers = []
        for i in range(num_tables):
            self.lookup_layers.append(
                hps.LookupLayer(
                    model_name="context_stream",
                    table_id=i,
                    emb_vec_size=EMB_VEC_SIZE,
                    emb_vec_dtype=EMB_VEC_DTYPE,
                    name="embedding_lookup" + str(i),
                )
            )
        self.fc = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer="ones",
            bias_initializer="zeros",
            name="fc",
        )

    def call(self, inputs):
        assert len(inputs) == len(self.lookup_layers)
        embeddings = []
        for i in range(len(inputs)):
            embeddings.append(
                tf.reshape(
                    self.lookup_layers[i](inputs[i]), shape=[-1, NUM_QUERY_KEY * EMB_VEC_SIZE]
                )
            )
        concat_embeddings = tf.concat(embeddings, axis=1)
        logit = self.fc(concat_embeddings)
        return logit

    def summary(self):
        inputs = []
        for _ in range(len(self.lookup_layers)):
            inputs.append(tf.keras.Input(shape=(NUM_QUERY_KEY,), dtype=TF_KEY_TYPE))
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()


def test_context_stream():
    set_up_model_files(NUM_TABLES, USE_CONTEXT_STREAM)
    model = InferenceModel(NUM_TABLES)
    model.summary()
    inputs_seq = []
    for _ in range(NUM_ITERS + 1):
        inputs = []
        for _ in range(NUM_TABLES):
            inputs.append(
                np.random.randint(0, VOCAB_SIZE, (MAX_BATCH_SIZE, NUM_QUERY_KEY)).astype(np.int64)
            )
        inputs_seq.append(inputs)

    hps.Init(ps_config_file="hps_for_context_stream.json", global_batch_size=MAX_BATCH_SIZE)
    preds = model(inputs_seq[0])

    start = time.time()
    for i in range(NUM_ITERS):
        preds = model(inputs_seq[i + 1])
    end = time.time()
    print("[INFO] Use context stream: ", hps_config["models"][0]["use_context_stream"])
    print(
        "[INFO] Elapsed time for "
        + str(NUM_ITERS)
        + " iterations: "
        + str(end - start)
        + " seconds"
    )

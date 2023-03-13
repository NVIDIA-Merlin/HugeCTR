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

hps_config = {
    "supportlonglong": False,
    "fuse_embedding_table": True,
    "models": [
        {
            "model": str(NUM_TABLES) + "_table",
            "sparse_files": [],
            "num_of_worker_buffer_in_pool": NUM_TABLES,
            "embedding_table_names": [],
            "embedding_vecsize_per_table": [],
            "maxnum_catfeature_query_per_table_per_sample": [],
            "default_value_for_each_table": [0.0],
            "deployed_device_list": [0],
            "max_batch_size": MAX_BATCH_SIZE,
            "cache_refresh_percentage_per_iteration": 1.0,
            "hit_rate_threshold": 1.0,
            "gpucacheper": 1.0,
            "gpucache": True,
            "use_static_table": False,
            "use_context_stream": True,
        }
    ],
}


def generate_embedding_tables(hugectr_sparse_model, vocab_range, embedding_vec_size):
    os.system("mkdir -p {}".format(hugectr_sparse_model))
    with open("{}/key".format(hugectr_sparse_model), "wb") as key_file, open(
        "{}/emb_vector".format(hugectr_sparse_model), "wb"
    ) as vec_file:
        for key in range(vocab_range[0], vocab_range[1]):
            vec = 0.00025 * np.ones((embedding_vec_size,)).astype(np.float32)
            key_struct = struct.pack("q", key)
            vec_struct = struct.pack(str(embedding_vec_size) + "f", *vec)
            key_file.write(key_struct)
            vec_file.write(vec_struct)


def set_up_model_files():
    for i in range(NUM_TABLES):
        table_name = "table" + str(i)
        model_file_name = "embeddings/" + table_name
        generate_embedding_tables(
            model_file_name, [i * VOCAB_SIZE, (i + 1) * VOCAB_SIZE], EMB_VEC_SIZE
        )
        hps_config["models"][0]["sparse_files"].append(model_file_name)
        hps_config["models"][0]["embedding_table_names"].append(table_name)
        hps_config["models"][0]["embedding_vecsize_per_table"].append(EMB_VEC_SIZE)
        hps_config["models"][0]["maxnum_catfeature_query_per_table_per_sample"].append(
            NUM_QUERY_KEY
        )
    return hps_config


class InferenceModel(tf.keras.models.Model):
    def __init__(self, num_tables, **kwargs):
        super(InferenceModel, self).__init__(**kwargs)
        self.lookup_layers = []
        for i in range(num_tables):
            self.lookup_layers.append(
                hps.LookupLayer(
                    model_name=str(NUM_TABLES) + "_table",
                    table_id=i,
                    emb_vec_size=EMB_VEC_SIZE,
                    emb_vec_dtype=EMB_VEC_DTYPE,
                    ps_config_file=str(NUM_TABLES) + "_table.json",
                    global_batch_size=MAX_BATCH_SIZE,
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


def create_savedmodel(hps_config):
    # Overwrite JSON configuration file
    hps_config["fuse_embedding_table"] = False
    hps_config_json_object = json.dumps(hps_config, indent=4)
    with open(str(NUM_TABLES) + "_table.json", "w") as outfile:
        outfile.write(hps_config_json_object)

    model = InferenceModel(NUM_TABLES)
    model.summary()
    inputs = []
    for i in range(NUM_TABLES):
        inputs.append(
            np.random.randint(
                i * VOCAB_SIZE, (i + 1) * VOCAB_SIZE, (MAX_BATCH_SIZE, NUM_QUERY_KEY)
            ).astype(np.int32)
        )
    model(inputs)
    model.save(str(NUM_TABLES) + "_table.savedmodel")

    # Overwrite JSON configuration file
    hps_config["fuse_embedding_table"] = True
    hps_config_json_object = json.dumps(hps_config, indent=4)
    with open(str(NUM_TABLES) + "_table.json", "w") as outfile:
        outfile.write(hps_config_json_object)


if __name__ == "__main__":
    hps_config = set_up_model_files()
    create_savedmodel(hps_config)

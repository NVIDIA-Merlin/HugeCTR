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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = dict()
args["ps_config_file"] = "hps.json"
args["global_batch_size"] = 1024
args["max_vocabulary_size_per_table_per_model"] = {"foo": [30000], "bar": [50000, 2000]}
args["num_iters"] = 10

hps_config = {
    "supportlonglong": False,
    "models": [
        {
            "model": "foo",
            "sparse_files": ["foo_sparse.model"],
            "num_of_worker_buffer_in_pool": 3,
            "embedding_table_names": ["sparse_embedding0"],
            "embedding_vecsize_per_table": [16],
            "maxnum_catfeature_query_per_table_per_sample": [10],
            "default_value_for_each_table": [1.0],
            "deployed_device_list": [0],
            "max_batch_size": 16384,
            "cache_refresh_percentage_per_iteration": 0.2,
            "hit_rate_threshold": 1.0,
            "gpucacheper": 1.0,
            "gpucache": True,
        },
        {
            "model": "bar",
            "sparse_files": ["bar_sparse_0.model", "bar_sparse_1.model"],
            "num_of_worker_buffer_in_pool": 3,
            "embedding_table_names": ["sparse_embedding0", "sparse_embedding1"],
            "embedding_vecsize_per_table": [64, 32],
            "maxnum_catfeature_query_per_table_per_sample": [3, 5],
            "default_value_for_each_table": [1.0, 1.0],
            "deployed_device_list": [0],
            "max_batch_size": 1024,
            "cache_refresh_percentage_per_iteration": 0.2,
            "hit_rate_threshold": 1.0,
            "gpucacheper": 1.0,
            "gpucache": True,
        },
    ],
}
hps_config_json_object = json.dumps(hps_config, indent=4)
with open(args["ps_config_file"], "w") as outfile:
    outfile.write(hps_config_json_object)


def convert_to_sparse_model(embeddings_weights, embedding_table_path, embedding_vec_size):
    os.system("mkdir -p {}".format(embedding_table_path))
    with open("{}/key".format(embedding_table_path), "wb") as key_file, open(
        "{}/emb_vector".format(embedding_table_path), "wb"
    ) as vec_file:
        for key in range(embeddings_weights.shape[0]):
            vec = embeddings_weights[key]
            key_struct = struct.pack("q", key)
            vec_struct = struct.pack(str(embedding_vec_size) + "f", *vec)
            key_file.write(key_struct)
            vec_file.write(vec_struct)


def _generate_embedding_tables():
    embedding_tables = {}
    for model in hps_config["models"]:
        model_name = model["model"]
        max_vocabulary_size_per_table = args["max_vocabulary_size_per_table_per_model"][model_name]
        embedding_vecsize_per_table = model["embedding_vecsize_per_table"]
        embedding_table_paths = model["sparse_files"]
        for table_id in range(len(max_vocabulary_size_per_table)):
            embedding_weights = np.random.random(
                (max_vocabulary_size_per_table[table_id], embedding_vecsize_per_table[table_id])
            ).astype(np.float32)
            convert_to_sparse_model(
                embedding_weights,
                embedding_table_paths[table_id],
                embedding_vecsize_per_table[table_id],
            )
            if model_name not in embedding_tables:
                embedding_tables[model_name] = [embedding_weights]
            else:
                embedding_tables[model_name].append(embedding_weights)
    return embedding_tables


def _generate_dense_keys(num_samples, vocabulary_range, max_nnz, key_dtype=np.int32):
    dense_keys = list()
    dense_keys = np.random.randint(
        low=vocabulary_range[0],
        high=vocabulary_range[1],
        size=(num_samples, max_nnz),
        dtype=key_dtype,
    )
    return dense_keys


def _generate_sparse_keys(num_samples, vocabulary_range, max_nnz, key_dtype=np.int32):
    indices = []
    values = []
    for i in range(num_samples):
        nnz = np.random.randint(low=1, high=max_nnz + 1)
        entries = sorted(np.random.choice(max_nnz, nnz, replace=False))
        for entry in entries:
            indices.append([i, entry])
        values.extend(
            np.random.randint(low=vocabulary_range[0], high=vocabulary_range[1], size=(nnz,))
        )
    values = np.array(values, dtype=key_dtype)
    return tf.sparse.SparseTensor(
        indices=indices, values=values, dense_shape=(num_samples, max_nnz)
    )


class TestHPS:
    embedding_tables = _generate_embedding_tables()

    def test_init(self):
        status = hps.Init(
            ps_config_file=args["ps_config_file"], global_batch_size=args["global_batch_size"]
        )
        assert "OK" == status

    def test_lookup_layer(cls):
        for model in hps_config["models"]:
            model_name = model["model"]
            embedding_vecsize_per_table = model["embedding_vecsize_per_table"]
            max_vocabulary_size_per_table = args["max_vocabulary_size_per_table_per_model"][
                model_name
            ]
            max_nnz_per_sample_per_table = model["maxnum_catfeature_query_per_table_per_sample"]
            for table_id in range(len(embedding_vecsize_per_table)):
                for max_norm in [None, tf.Variable(0.0), tf.Variable(0.1), tf.Variable(10.0)]:
                    lookup_layer = hps.LookupLayer(
                        model_name=model_name,
                        table_id=table_id,
                        emb_vec_size=embedding_vecsize_per_table[table_id],
                        emb_vec_dtype=tf.float32,
                    )
                    for i in range(args["num_iters"]):
                        dense_keys = _generate_dense_keys(
                            args["global_batch_size"],
                            [0, max_vocabulary_size_per_table[table_id]],
                            max_nnz_per_sample_per_table[table_id],
                        )
                        embeddings = lookup_layer(ids=dense_keys, max_norm=max_norm)
                        embeddings_gt = tf.nn.embedding_lookup(
                            params=cls.embedding_tables[model_name][table_id],
                            ids=dense_keys,
                            max_norm=max_norm,
                        )
                        flag = tf.reduce_all(tf.equal(embeddings, embeddings_gt))
                        assert True == flag

    def test_sparse_lookup_layer(cls):
        for model in hps_config["models"]:
            model_name = model["model"]
            embedding_vecsize_per_table = model["embedding_vecsize_per_table"]
            max_vocabulary_size_per_table = args["max_vocabulary_size_per_table_per_model"][
                model_name
            ]
            max_nnz_per_sample_per_table = model["maxnum_catfeature_query_per_table_per_sample"]
            for table_id in range(len(embedding_vecsize_per_table)):
                for combiner in ["sum", "mean"]:
                    for max_norm in [None, tf.Variable(0.0), tf.Variable(0.1), tf.Variable(10.0)]:
                        sparse_lookup_layer = hps.SparseLookupLayer(
                            model_name=model_name,
                            table_id=table_id,
                            emb_vec_size=embedding_vecsize_per_table[table_id],
                            emb_vec_dtype=tf.float32,
                        )
                        for i in range(args["num_iters"]):
                            sparse_keys = _generate_sparse_keys(
                                args["global_batch_size"],
                                [0, max_vocabulary_size_per_table[table_id]],
                                max_nnz_per_sample_per_table[table_id],
                            )
                            embeddings = sparse_lookup_layer(
                                sp_ids=sparse_keys,
                                sp_weights=None,
                                combiner=combiner,
                                max_norm=max_norm,
                            )
                            embeddings_gt = tf.nn.embedding_lookup_sparse(
                                params=cls.embedding_tables[model_name][table_id],
                                sp_ids=sparse_keys,
                                sp_weights=None,
                                combiner=combiner,
                                max_norm=max_norm,
                            )
                            flag = tf.reduce_all(tf.equal(embeddings, embeddings_gt))
                            assert True == flag

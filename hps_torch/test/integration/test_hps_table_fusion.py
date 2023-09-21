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

import torch
import hps_torch
from typing import List
import os
import numpy as np
import struct
import json
import pytest
import time

NUM_GPUS = 1
VOCAB_SIZE = 10000
EMB_VEC_SIZE = 128
NUM_QUERY_KEY = 10
MAX_BATCH_SIZE = 256
NUM_ITERS = 100
NUM_TABLES = 8
USE_CONTEXT_STREAM = True

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(NUM_GPUS)))

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
            "embedding_cache_type": "static",
            "use_context_stream": True,
        }
    ],
}


class Model(torch.nn.Module):
    def __init__(self, ps_config_file: str, model_name: str, emb_vec_size: List[int]):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                hps_torch.LookupLayer(ps_config_file, model_name, table_id, emb_vec_size[table_id])
                for table_id in range(len(emb_vec_size))
            ]
        )

    def forward(self, keys_list: torch.Tensor):
        vectors = []
        futures = torch.jit.annotate(List[torch.jit.Future[torch.Tensor]], [])
        for i, layer in enumerate(self.layers):
            fut = torch.jit.fork(layer, keys_list[i])
            futures.append(fut)
        for i, _ in enumerate(self.layers):
            vectors.append(torch.jit.wait(futures[i]))
        return torch.cat(vectors)


def generate_embedding_tables(
    hugectr_sparse_model, vocab_range, embedding_vec_size, embedding_table
):
    os.system("mkdir -p {}".format(hugectr_sparse_model))
    with open("{}/key".format(hugectr_sparse_model), "wb") as key_file, open(
        "{}/emb_vector".format(hugectr_sparse_model), "wb"
    ) as vec_file:
        for key in range(vocab_range[0], vocab_range[1]):
            vec = np.random.random((embedding_vec_size,)).astype(np.float32)
            key_struct = struct.pack("q", key)
            vec_struct = struct.pack(str(embedding_vec_size) + "f", *vec)
            key_file.write(key_struct)
            vec_file.write(vec_struct)
            embedding_table[key] = vec


def set_up_model_files():
    embedding_table = np.zeros((NUM_TABLES * VOCAB_SIZE, EMB_VEC_SIZE)).astype(np.float32)
    for i in range(NUM_TABLES):
        table_name = "table" + str(i)
        model_file_name = "embeddings/" + table_name
        generate_embedding_tables(
            model_file_name, [i * VOCAB_SIZE, (i + 1) * VOCAB_SIZE], EMB_VEC_SIZE, embedding_table
        )
        hps_config["models"][0]["sparse_files"].append(model_file_name)
        hps_config["models"][0]["embedding_table_names"].append(table_name)
        hps_config["models"][0]["embedding_vecsize_per_table"].append(EMB_VEC_SIZE)
        hps_config["models"][0]["maxnum_catfeature_query_per_table_per_sample"].append(
            NUM_QUERY_KEY
        )
    hps_config_json_object = json.dumps(hps_config, indent=4)
    with open(str(NUM_TABLES) + "_table.json", "w") as outfile:
        outfile.write(hps_config_json_object)
    return embedding_table


def test_hps_table_fusion():
    embedding_table = set_up_model_files()
    model = torch.jit.script(
        Model(
            f"{NUM_TABLES}_table.json",
            f"{NUM_TABLES}_table",
            [EMB_VEC_SIZE for _ in range(NUM_TABLES)],
        )
    )
    inputs_seq = []
    for _ in range(NUM_ITERS + 1):
        inputs = []
        for i in range(NUM_TABLES):
            inputs.append(
                torch.randint(
                    i * VOCAB_SIZE,
                    (i + 1) * VOCAB_SIZE,
                    (MAX_BATCH_SIZE, NUM_QUERY_KEY),
                    dtype=torch.int32,
                ).cuda()
            )
        inputs_seq.append(torch.stack(inputs))

    preds = model(inputs_seq[0])
    preds_seq = []
    start = time.time()
    for i in range(NUM_ITERS):
        preds_seq.append(model(inputs_seq[i + 1]))
    end = time.time()
    print(
        "[INFO] Elapsed time for "
        + str(NUM_ITERS)
        + " iterations: "
        + str(end - start)
        + " seconds"
    )
    preds_seq = torch.stack(preds_seq).cpu().numpy()

    preds_seq_gt = []
    for i in range(NUM_ITERS):
        preds_seq_gt.append(np.concatenate(embedding_table[inputs_seq[i + 1].cpu().numpy()]))
    preds_seq_gt = np.array(preds_seq_gt)

    diff = preds_seq - preds_seq_gt
    mse = np.mean(diff * diff)
    assert mse <= 1e-6
    print(f"HPS Torch Plugin embedding lookup with table fusion, MSE: {mse} ")

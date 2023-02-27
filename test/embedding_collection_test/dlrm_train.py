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

import hugectr
from mpi4py import MPI


def generate_shard_plan(slot_size_array, num_gpus):
    mp_table = [i for i in range(len(slot_size_array)) if slot_size_array[i] > 6000]
    dp_table = [i for i in range(len(slot_size_array)) if slot_size_array[i] <= 6000]
    shard_matrix = [[] for _ in range(num_gpus)]
    shard_strategy = [("mp", [str(i) for i in mp_table]), ("dp", [str(i) for i in dp_table])]

    for table_id in dp_table:
        for gpu_id in range(num_gpus):
            shard_matrix[gpu_id].append(str(table_id))

    for i, table_id in enumerate(mp_table):
        target_gpu = i % num_gpus
        shard_matrix[target_gpu].append(str(table_id))
    return shard_matrix, shard_strategy


solver = hugectr.CreateSolver(
    max_eval_batches=70,
    batchsize_eval=65536,
    batchsize=65536,
    lr=0.5,
    warmup_steps=300,
    vvgpu=[[0, 1, 2, 3, 4, 5, 6, 7]],
    repeat_dataset=True,
    i64_input_key=True,
    metrics_spec={hugectr.MetricsType.AverageLoss: 0.0},
    use_embedding_collection=True,
)
slot_size_array = [
    203931,
    18598,
    14092,
    7012,
    18977,
    4,
    6385,
    1245,
    49,
    186213,
    71328,
    67288,
    11,
    2168,
    7338,
    61,
    4,
    932,
    15,
    204515,
    141526,
    199433,
    60919,
    9137,
    71,
    34,
]
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.Parquet,
    source=["./criteo_data/train/_file_list.txt"],
    eval_source="./criteo_data/val/_file_list.txt",
    check_type=hugectr.Check_t.Non,
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.SGD, update_type=hugectr.Update_t.Local, atomic_update=True
)
model = hugectr.Model(solver, reader, optimizer)

num_embedding = 26

model.add(
    hugectr.Input(
        label_dim=1,
        label_name="label",
        dense_dim=13,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("data{}".format(i), 1, False, 1)
            for i in range(num_embedding)
        ],
    )
)

# create embedding table
embedding_table_list = []
for i in range(num_embedding):
    embedding_table_list.append(
        hugectr.EmbeddingTableConfig(
            name=str(i), max_vocabulary_size=slot_size_array[i], ev_size=128
        )
    )
# create ebc config
ebc_config = hugectr.EmbeddingCollectionConfig()
emb_vec_list = []
for i in range(num_embedding):
    ebc_config.embedding_lookup(
        table_config=embedding_table_list[i],
        bottom_name="data{}".format(i),
        top_name="emb_vec{}".format(i),
        combiner="sum",
    )
shard_matrix, shard_strategy = generate_shard_plan(slot_size_array, 8)
ebc_config.shard(shard_matrix=shard_matrix, shard_strategy=shard_strategy)

model.add(ebc_config)
# need concat
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Concat,
        bottom_names=["emb_vec{}".format(i) for i in range(num_embedding)],
        top_names=["sparse_embedding1"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["dense"],
        top_names=["fc1"],
        num_output=512,
    )
)

model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"])
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu1"],
        top_names=["fc2"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu2"],
        top_names=["fc3"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc3"], top_names=["relu3"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Interaction,  # interaction only support 3-D input
        bottom_names=["relu3", "sparse_embedding1"],
        top_names=["interaction1"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["interaction1"],
        top_names=["fc4"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc4"], top_names=["relu4"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu4"],
        top_names=["fc5"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc5"], top_names=["relu5"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu5"],
        top_names=["fc6"],
        num_output=512,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc6"], top_names=["relu6"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu6"],
        top_names=["fc7"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc7"], top_names=["relu7"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu7"],
        top_names=["fc8"],
        num_output=1,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["fc8", "label"],
        top_names=["loss"],
    )
)
model.compile()
model.summary()
model.fit(max_iter=1000, display=100, eval_interval=100, snapshot=10000000, snapshot_prefix="dlrm")

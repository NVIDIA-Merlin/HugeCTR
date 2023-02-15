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

# from mpi4py import MPI


def generate_shard_plan(slot_size_array, num_gpus):
    mp_table = [i for i in range(len(slot_size_array))]
    shard_matrix = [[] for _ in range(num_gpus)]
    shard_strategy = [("mp", [str(i) for i in mp_table])]

    for i, table_id in enumerate(mp_table):
        target_gpu = i % num_gpus
        shard_matrix[target_gpu].append(str(table_id))
    return shard_matrix, shard_strategy


# 1. Create Solver, DataReaderParams and Optimizer
solver = hugectr.CreateSolver(
    max_eval_batches=51,
    batchsize_eval=16384,  # 1769472,
    batchsize=8192,  # 55296,
    vvgpu=[[0, 1, 2, 3, 4, 5, 6, 7]],
    repeat_dataset=True,
    lr=24.0,
    warmup_steps=2750,
    decay_start=49315,
    decay_steps=27772,
    decay_power=2.0,
    end_lr=0.0,
    use_mixed_precision=True,
    scaler=1024,
    use_cuda_graph=False,
    async_mlp_wgrad=False,
    gen_loss_summary=False,
    train_intra_iteration_overlap=False,
    train_inter_iteration_overlap=False,
    eval_intra_iteration_overlap=False,  # doesn't do anything
    eval_inter_iteration_overlap=False,
    all_reduce_algo=hugectr.AllReduceAlgo.NCCL,
    grouped_all_reduce=False,
    num_iterations_statistics=20,
    metrics_spec={hugectr.MetricsType.AUC: 0.8025},
    perf_logging=True,
    drop_incomplete_batch=True,  # False,
    use_embedding_collection=True,
)

"""
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.Raw,
    source=["/criteo/mlperf/train_data.bin"],
    eval_source="/criteo/mlperf/test_data.bin",
    check_type=hugectr.Check_t.Non,
    num_samples=4195197692,
    eval_num_samples=89137319,
    cache_eval_data=51,
)"""

optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.SGD, update_type=hugectr.Update_t.Local, atomic_update=True
)

table_size_array = [
    40000000,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    40000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    40000000,
    40000000,
    40000000,
    590152,
    12973,
    108,
    36,
]
multi_hot_sizes = [
    3,
    2,
    1,
    2,
    6,
    1,
    1,
    1,
    1,
    7,
    3,
    8,
    1,
    6,
    9,
    5,
    1,
    1,
    1,
    12,
    100,
    27,
    10,
    3,
    1,
    1,
]


num_table = len(table_size_array)

reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync,
    source=[
        "/lustre/fsw/mlperf/mlperft-dlrm/jlasek/datasets/criteo_multihot_subset/train_data.bin"
    ],
    eval_source="/lustre/fsw/mlperf/mlperft-dlrm/jlasek/datasets/criteo_multihot_subset/test_data.bin",
    check_type=hugectr.Check_t.Non,
    num_samples=4195197692,
    eval_num_samples=89137319,
    cache_eval_data=1,
    slot_size_array=table_size_array,
    async_param=hugectr.AsyncParam(
        multi_hot_reader=True, num_threads=1, num_batches_per_thread=16, shuffle=False
    ),
)

# 2. Initialize the Model instance
model = hugectr.Model(solver, reader, optimizer)
# 3. Construct the Model graph
model.add(
    hugectr.Input(
        label_dim=1,
        label_name="label",
        dense_dim=13,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("data{}".format(i), multi_hot_sizes[i], True, 1)
            for i in range(num_table)
        ],
    )
)

# create embedding table
embedding_table_list = []
for i in range(num_table):
    embedding_table_list.append(
        hugectr.EmbeddingTableConfig(
            name=str(i), max_vocabulary_size=table_size_array[i], ev_size=128
        )
    )
# create embedding planner and embedding collection
ebc_config = hugectr.EmbeddingCollectionConfig()
emb_vec_list = []
for i in range(num_table):
    ebc_config.embedding_lookup(
        table_config=embedding_table_list[i],
        bottom_name="data{}".format(i),
        top_name="emb_vec{}".format(i),
        combiner="sum",
    )
shard_matrix, shard_strategy = generate_shard_plan(table_size_array, 8)
print("shard_matrix", shard_matrix)
ebc_config.shard(shard_matrix=shard_matrix, shard_strategy=shard_strategy)

model.add(ebc_config)
# need concat
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Concat,
        bottom_names=["emb_vec{}".format(i) for i in range(num_table)],
        top_names=["sparse_embedding1"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Head,
        bottom_names=["dense"],
        top_names=["fc11", "fc12", "fc13", "fc14"],
        num_output=512,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Body,
        bottom_names=["fc11", "fc12", "fc13", "fc14"],
        top_names=["fc21", "fc22", "fc23", "fc24"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Tail,
        bottom_names=["fc21", "fc22", "fc23", "fc24"],
        top_names=["fc3"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Interaction,
        bottom_names=["fc3", "sparse_embedding1"],
        top_names=["interaction1", "interaction_grad"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Head,
        bottom_names=["interaction1", "interaction_grad"],
        top_names=["fc41", "fc42", "fc43", "fc44"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Body,
        bottom_names=["fc41", "fc42", "fc43", "fc44"],
        top_names=["fc51", "fc52", "fc53", "fc54"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Body,
        bottom_names=["fc51", "fc52", "fc53", "fc54"],
        top_names=["fc61", "fc62", "fc63", "fc64"],
        num_output=512,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Body,
        bottom_names=["fc61", "fc62", "fc63", "fc64"],
        top_names=["fc71", "fc72", "fc73", "fc74"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Tail,
        act_type=hugectr.Activation_t.Non,
        bottom_names=["fc71", "fc72", "fc73", "fc74"],
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
# 4. Dump the Model graph to JSON
model.graph_to_json(graph_config_file="dlrm.json")
# 5. Compile & Fit
model.compile()
model.summary()
model.fit(
    max_iter=75868, display=1000, eval_interval=3793, snapshot=2000000, snapshot_prefix="dlrm"
)

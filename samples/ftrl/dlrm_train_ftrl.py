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
import argparse
from mpi4py import MPI

parser = argparse.ArgumentParser(
    description="HugeCTR Embedding Collection DLRM model training script."
)
parser.add_argument(
    "--use_mixed_precision",
    action="store_true",
)
parser.add_argument(
    "--shard_plan",
    help="shard strategy",
    type=str,
    choices=["round_robin", "uniform", "hybrid"],
)
parser.add_argument(
    "--use_dynamic_hash_table",
    action="store_true",
)
parser.add_argument(
    "--optimizer",
    help="Optimizer to use",
    type=str,
    choices=["ftrl", "sgd"],
    default="sgd",
)
parser.add_argument(
    "--beta",
    help="beta value for Ftrl",
    type=float,
    default=0.9,
)
parser.add_argument(
    "--lambda1",
    help="lambda1 value for Ftrl",
    type=float,
    default=0.1,
)
parser.add_argument(
    "--lambda2",
    help="lambda1 value for Ftrl",
    type=float,
    default=0.1,
)

args = parser.parse_args()

comm = MPI.COMM_WORLD
num_nodes = comm.Get_size()
rank = comm.Get_rank()
num_gpus_per_node = 8
num_gpus = num_nodes * num_gpus_per_node


def generate_shard_plan(slot_size_array, num_gpus):
    if args.shard_plan == "round_robin":
        shard_strategy = [("mp", [str(i) for i in range(len(slot_size_array))])]
        shard_matrix = [[] for _ in range(num_gpus)]
        for i, table_id in enumerate(range(len(slot_size_array))):
            target_gpu = i % num_gpus
            shard_matrix[target_gpu].append(str(table_id))
    elif args.shard_plan == "uniform":
        shard_strategy = [("mp", [str(i) for i in range(len(slot_size_array))])]
        shard_matrix = [[] for _ in range(num_gpus)]
        for table_id in range(len(slot_size_array)):
            for gpu_id in range(num_gpus):
                shard_matrix[gpu_id].append(str(table_id))
    elif args.shard_plan == "hybrid":
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
    else:
        raise Exception(args.shard_plan + " is not supported")
    return shard_matrix, shard_strategy


solver = hugectr.CreateSolver(
    max_eval_batches=70,
    batchsize_eval=65536,
    batchsize=65536,
    lr=0.5,
    use_mixed_precision=args.use_mixed_precision,
    warmup_steps=300,
    vvgpu=[[x for x in range(num_gpus_per_node)]] * num_nodes,
    repeat_dataset=True,
    i64_input_key=True,
    metrics_spec={hugectr.MetricsType.AverageLoss: 0.0},
    use_embedding_collection=True,
)
slot_size_array = [
    30,
    1000,
    6000,
    6500,
    52000,
    200000,
    200000,
    240000,
    440000,
    10,
    5,
    2,
    1,
    100,
    1500000,
    200,
    70000,
    200000,
    110000,
    550000,
    120000,
    20000,
    125000,
    50000,
    50000,
    20,
    5,
    100000,
    20000,
    800000,
    60,
    400,
    120,
    4,
    1000,
    140000,
    5,
    50000,
    40000,
    5000,
    2000,
    7000,
    15000,
    1000,
    50,
    300,
    5000,
    30000,
]

reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync,
    source=["./bing_proxy_raw_small/train.bin"],
    eval_source="./bing_proxy_raw_small/train.bin",
    check_type=hugectr.Check_t.Non,
    num_samples=20000000,
    eval_num_samples=1000000,
    cache_eval_data=1,
    slot_size_array=slot_size_array,
    async_param=hugectr.AsyncParam(
        num_threads=1,
        num_batches_per_thread=16,
        shuffle=False,
        multi_hot_reader=True,
        is_dense_float=True,
    ),
)
optimizer = None
if args.optimizer == "ftrl":
    optimizer = hugectr.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.Ftrl,
        update_type=hugectr.Update_t.Global,
        beta=args.beta,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
    )
elif args.optimizer == "sgd":
    optimizer = hugectr.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.SGD,
        update_type=hugectr.Update_t.Local,
        atomic_update=True,
    )

model = hugectr.Model(solver, reader, optimizer)

num_embedding = 48

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
            name=str(i),
            max_vocabulary_size=-1 if args.use_dynamic_hash_table else slot_size_array[i],
            ev_size=128,
        )
    )
# create ebc config
ebc_config = hugectr.EmbeddingCollectionConfig(use_exclusive_keys=True)
emb_vec_list = []
for i in range(num_embedding):
    ebc_config.embedding_lookup(
        table_config=embedding_table_list[i],
        bottom_name="data{}".format(i),
        top_name="emb_vec{}".format(i),
        combiner="sum",
    )
shard_matrix, shard_strategy = generate_shard_plan(slot_size_array, num_gpus)
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

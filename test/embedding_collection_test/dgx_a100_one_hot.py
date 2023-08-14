# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import math

from mpi4py import MPI

import hugectr

TRAIN_NUM_SAMPLES = 4195197692
EVAL_NUM_SAMPLES = 89137319
TABLE_SIZE_ARRAY = [
    39884406,
    39043,
    17289,
    7420,
    20263,
    3,
    7120,
    1543,
    63,
    38532951,
    2953546,
    403346,
    10,
    2208,
    11938,
    155,
    4,
    976,
    14,
    39979771,
    25641295,
    39664984,
    585935,
    12972,
    108,
    36,
]

NUM_TABLE = len(TABLE_SIZE_ARRAY)

MULTI_HOT_SIZES = [1 for _ in range(NUM_TABLE)]

NUM_DENSE = 13

parser = argparse.ArgumentParser(description="HugeCTR DLRM V2 model training script.")
parser.add_argument(
    "--batchsize",
    help="Batch size for training",
    type=int,
    default=8192,
)
parser.add_argument(
    "--batchsize_eval",
    help="Batch size for evaluation",
    type=int,
    default=16384,
)
parser.add_argument(
    "--max_eval_batches",
    help="The number of evaluation batches to use",
    type=int,
    default=None,
)
parser.add_argument(
    "--lr",
    help="Learning rate",
    type=float,
    default=24.0,
)
parser.add_argument(
    "--warmup_steps",
    help="Warmup steps",
    type=int,
    default=2750,
)
parser.add_argument(
    "--decay_start",
    help="Decay start",
    type=int,
    default=49315,
)
parser.add_argument(
    "--decay_steps",
    help="Decay steps",
    type=int,
    default=27772,
)
parser.add_argument(
    "--use_mixed_precision",
    action="store_true",
)
parser.add_argument(
    "--scaler",
    help="Loss scaling constant",
    type=float,
    default=1024,
)
parser.add_argument(
    "--enable_tf32_compute",
    action="store_true",
)
parser.add_argument(
    "--disable_algorithm_search",
    help="Disables GEMM algorithm search for fully connected layers",
    dest="use_algorithm_search",
    action="store_false",
)
parser.add_argument(
    "--gen_loss_summary",
    help="Compute loss summary during training (loss = 0 if not set)",
    action="store_true",
)
parser.add_argument(
    "--max_iter",
    help="Number of training iterations to run",
    type=int,
    default=None,
)
parser.add_argument(
    "--display_interval",
    help="Display throughput stats every number of iterations",
    type=int,
    default=1000,
)
parser.add_argument(
    "--eval_interval",
    help="Evaluate every number of iterations given",
    type=int,
    default=None,
)
parser.add_argument(
    "--auc_threshold",
    help="AUC threshold to reach to stop training",
    type=float,
    default=0.8025,
)
parser.add_argument(
    "--num_gpus_per_node",
    help="The number of GPUs per node",
    type=int,
    default=8,
)
parser.add_argument(
    "--ev_size",
    help="The width of the embedding vector",
    type=int,
    default=128,
)

args = parser.parse_args()
comm = MPI.COMM_WORLD
num_nodes = comm.Get_size()
rank = comm.Get_rank()
num_gpus = num_nodes * args.num_gpus_per_node
is_rank_zero = rank == 0

# Dependent parameters (if not set)
iter_per_epoch = TRAIN_NUM_SAMPLES / args.batchsize
if args.max_iter is None:
    args.max_iter = math.ceil(iter_per_epoch)
if args.eval_interval is None:
    args.eval_interval = math.floor(0.05 * iter_per_epoch)
if args.max_eval_batches is None:
    args.max_eval_batches = math.ceil(EVAL_NUM_SAMPLES / args.batchsize_eval)


def rowwise_sharding_plan():
    mp_table = [i for i in range(NUM_TABLE)]
    shard_matrix = [[] for _ in range(num_gpus)]
    shard_strategy = [("mp", [str(i) for i in mp_table])]

    for i, table_id in enumerate(mp_table):
        for gpu_id in range(num_gpus):
            shard_matrix[gpu_id].append(str(table_id))
    return shard_matrix, shard_strategy


shard_matrix, shard_strategy = rowwise_sharding_plan()

# 1. Create Solver, DataReaderParams and Optimizer
solver = hugectr.CreateSolver(
    model_name="DLRM",
    max_eval_batches=args.max_eval_batches,
    batchsize_eval=args.batchsize_eval,
    batchsize=args.batchsize,
    vvgpu=[[x for x in range(args.num_gpus_per_node)] for _ in range(num_nodes)],
    repeat_dataset=True,
    lr=args.lr,
    warmup_steps=args.warmup_steps,
    decay_start=args.decay_start,
    decay_steps=args.decay_steps,
    decay_power=2.0,
    end_lr=0.0,
    use_mixed_precision=args.use_mixed_precision,
    enable_tf32_compute=args.enable_tf32_compute,
    scaler=args.scaler,
    use_cuda_graph=True,
    gen_loss_summary=args.gen_loss_summary,
    train_intra_iteration_overlap=True,
    train_inter_iteration_overlap=True,
    eval_intra_iteration_overlap=False,
    eval_inter_iteration_overlap=False,
    all_reduce_algo=hugectr.AllReduceAlgo.NCCL,
    grouped_all_reduce=True,
    num_iterations_statistics=20,
    metrics_spec={hugectr.MetricsType.AUC: args.auc_threshold},
    perf_logging=False,
    drop_incomplete_batch=True,
    use_embedding_collection=True,
    use_algorithm_search=args.use_algorithm_search,
)

optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.SGD,
    update_type=hugectr.Update_t.Local,
    atomic_update=True,
)

reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync,
    source=["/data/train_data.bin"],
    eval_source="/data/test_data.bin",
    check_type=hugectr.Check_t.Non,
    num_samples=TRAIN_NUM_SAMPLES,
    eval_num_samples=EVAL_NUM_SAMPLES,
    cache_eval_data=1,
    slot_size_array=TABLE_SIZE_ARRAY,
    async_param=hugectr.AsyncParam(
        num_threads=1,
        num_batches_per_thread=16,
        shuffle=False,
        multi_hot_reader=True,
        is_dense_float=True,
    ),
)

# 2. Initialize the Model instance
model = hugectr.Model(solver, reader, optimizer)
# 3. Construct the Model graph
model.add(
    hugectr.Input(
        label_dim=1,
        label_name="label",
        dense_dim=NUM_DENSE,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("data{}".format(i), MULTI_HOT_SIZES[i], True, 1)
            for i in range(NUM_TABLE)
        ],
    )
)

# create embedding table
embedding_table_list = []
for i in range(NUM_TABLE):
    embedding_table_list.append(
        hugectr.EmbeddingTableConfig(
            name=str(i), max_vocabulary_size=TABLE_SIZE_ARRAY[i], ev_size=args.ev_size
        )
    )
# create embedding planner and embedding collection
comm_strategy = (
    hugectr.CommunicationStrategy.Hierarchical
    if num_nodes > 1
    else hugectr.CommunicationStrategy.Uniform
)
ebc_config = hugectr.EmbeddingCollectionConfig(use_exclusive_keys=True, comm_strategy=comm_strategy)
ebc_config.embedding_lookup(
    table_config=[embedding_table_list[i] for i in range(NUM_TABLE)],
    bottom_name=["data{}".format(i) for i in range(NUM_TABLE)],
    top_name="sparse_embedding",
    combiner=["concat" for _ in range(NUM_TABLE)],
)

ebc_config.shard(shard_matrix=shard_matrix, shard_strategy=shard_strategy)

model.add(ebc_config)

# configure compute knobs for bottom & top MLP layers
compute_config = hugectr.DenseLayerComputeConfig(
    async_wgrad=True,
    fuse_wb=False,
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MLP,
        bottom_names=["dense"],
        top_names=["mlp1"],
        num_outputs=[512, 256, 128],
        act_type=hugectr.Activation_t.Relu,
        compute_config=compute_config,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["sparse_embedding"],
        top_names=["sparse_embedding1"],
        shape=[-1, NUM_TABLE, args.ev_size],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Interaction,
        bottom_names=["mlp1", "sparse_embedding1"],
        top_names=["interaction1"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MLP,
        bottom_names=["interaction1"],
        top_names=["mlp2"],
        num_outputs=[1024, 1024, 512, 256, 1],
        activations=[
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Non,
        ],
        compute_config=compute_config,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["mlp2", "label"],
        top_names=["loss"],
    )
)
# 4. Compile & Fit
model.compile()
model.summary()

model.fit(
    max_iter=args.max_iter,
    display=args.display_interval,
    eval_interval=args.eval_interval,
    snapshot=2000000,
    snapshot_prefix="dlrm",
)

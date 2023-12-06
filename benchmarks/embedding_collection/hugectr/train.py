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
import logging
import math

from mpi4py import MPI
import hugectr
import sharding
from typing import Iterator, List, Optional
import sys


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HugeCTR DCN V2 model training script.")
    parser.add_argument("--dataset_path", type=str, default="/data/train/bin")
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
        default=128,
    )
    parser.add_argument(
        "--num_gpus_per_node",
        help="The number of GPUs per node",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--train_num_samples",
        help="The number of train samples",
        type=int,
        default=300 * 65536,
    )
    parser.add_argument(
        "--eval_num_samples",
        help="The number of train samples",
        type=int,
        default=300 * 65536,
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
        default=200,
    )
    parser.add_argument(
        "--max_eval_batches",
        help="The number of evaluation batches to use",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--display_interval",
        help="Display throughput stats every number of iterations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--eval_interval",
        help="Evaluate every number of iterations given",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--i64_input_key",
        help="int64 key_type",
        action="store_true",
    )
    parser.add_argument("--perf_logging", action="store_true")

    # sharding
    parser.add_argument(
        "--sharding_plan",
        help="Sharding plan to use",
        type=str,
        choices=["round_robin", "uniform", "auto", "hier_auto", "table_row_wise"],
        default="round_robin",
    )
    parser.add_argument(
        "--mem_comm_bw_ratio",
        help="The ratio between the communication and the memory bw of the system",
        type=float,
        default=2000 / 25,
    )
    parser.add_argument(
        "--mem_comm_work_ratio",
        help="The ratio between the communication and the memory work of the network",
        type=float,
        default=8 / 2,
    )
    parser.add_argument(
        "--dense_comm_work_ratio",
        help="The ratio between the communication and the memory work of the network",
        type=float,
        default=4 / 2,
    )
    parser.add_argument(
        "--memory_cap_for_embedding",
        help="The amount of memory can be used for storing embedding in GB",
        type=float,
        default=60,
    )

    # embedding + mlp
    parser.add_argument(
        "--num_table",
        help="number of each type of embedding table, can be an integer or list of integer separated by comma",
        type=str,
        default="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
    )
    parser.add_argument(
        "--vocabulary_size_per_table",
        help="number of vocabulary_size of each type of embedding table, can be a list of integer separated by comma",
        type=str,
        default="40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36",
    )
    parser.add_argument(
        "--nnz_per_table",
        help="number of nnz of each type of embedding table, can be a list of integer separated by comma",
        type=str,
        default="3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1",
    )
    parser.add_argument(
        "--ev_size_per_table",
        help="ev_size of each type of embedding table, can be an integer or a list of integer separated by comma",
        type=str,
        default="128",
    )

    parser.add_argument(
        "--dense_dim",
        help="dense input dim",
        type=int,
        default=13,
    )
    parser.add_argument(
        "--bmlp_layer_sizes",
        type=str,
        default="512,256,128",
        help="Comma separated layer sizes for bottom mlp.",
    )
    parser.add_argument(
        "--tmlp_layer_sizes",
        type=str,
        default="1024,1024,512,256,1",
        help="Comma separated layer sizes for top mlp.",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="Number of DCN layers in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="Low rank dimension for DCN in interaction layer (only on dlrm with DCN).",
    )
    # optimizer
    parser.add_argument(
        "--optimizer",
        help="Optimizer to use",
        type=str,
        choices=["adagrad", "sgd"],
        default="sgd",
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--eps",
        help="Epsilon value for Adagrad",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--init_accu",
        help="Initial accumulator value for Adagrad",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--warmup_steps",
        help="Warmup steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--decay_start",
        help="Decay start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--decay_steps",
        help="Decay steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--auc_threshold",
        help="AUC threshold to reach to stop training",
        type=float,
        default=0.80275,
    )

    # optimization option
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
    )
    parser.add_argument(
        "--scaler",
        help="Loss scaling constant",
        type=float,
        default=1.0,
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
        "--disable_cuda_graph",
        action="store_false",
    )
    parser.add_argument(
        "--disable_train_intra_iteration_overlap",
        action="store_false",
    )
    parser.add_argument(
        "--disable_train_inter_iteration_overlap",
        action="store_false",
    )
    parser.add_argument(
        "--disable_fuse_sparse_embedding",
        action="store_false",
    )
    parser.add_argument(
        "--dp_threshold",
        help="select $dp_threshold table as dp sharding",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dense_threshold",
        help="select $dense_threshold table using dense compression",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_column_wise_shard",
        action="store_true",
    )

    args = parser.parse_args(argv)
    num_table_list = args.num_table.strip().split(",")
    vocabulary_size_per_table = args.vocabulary_size_per_table.strip().split(",")
    nnz_per_table = args.nnz_per_table.split(",")
    ev_size_per_table = args.ev_size_per_table.split(",")
    num_vocabulary_size = len(vocabulary_size_per_table)

    assert len(num_table_list) == num_vocabulary_size
    assert len(nnz_per_table) == num_vocabulary_size
    assert len(ev_size_per_table) == 1 or len(ev_size_per_table) == num_vocabulary_size
    if len(ev_size_per_table) == 1:
        ev_size_per_table = [ev_size_per_table[0] for _ in range(num_vocabulary_size)]
    num_table_list = [int(v) for v in num_table_list]

    args.TABLE_SIZE_ARRAY = []
    args.MULTI_HOT_SIZES = []
    args.EMB_VEC_SIZES = []
    for i, num_table in enumerate(num_table_list):
        for _ in range(num_table):
            args.TABLE_SIZE_ARRAY.append(int(vocabulary_size_per_table[i]))
            args.MULTI_HOT_SIZES.append(int(nnz_per_table[i]))
            args.EMB_VEC_SIZES.append(int(ev_size_per_table[i]))

    args.NUM_TABLE = len(args.TABLE_SIZE_ARRAY)
    args.NUM_DENSE = int(args.dense_dim)
    args.bmlp_layer_sizes = [int(v) for v in args.bmlp_layer_sizes.strip().split(",")]
    args.tmlp_layer_sizes = [int(v) for v in args.tmlp_layer_sizes.strip().split(",")]

    return args


args = parse_args(sys.argv[1:])

comm = MPI.COMM_WORLD
num_nodes = comm.Get_size()
rank = comm.Get_rank()
num_gpus = num_nodes * args.num_gpus_per_node
is_rank_zero = rank == 0
is_rank_zero = rank == 0

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Dependent parameters (if not set)
iter_per_epoch = args.train_num_samples / args.batchsize
if args.max_iter is None:
    args.max_iter = math.ceil(iter_per_epoch)
if args.eval_interval is None:
    args.eval_interval = math.floor(0.05 * iter_per_epoch)
if args.max_eval_batches is None:
    args.max_eval_batches = math.ceil(args.eval_num_samples / args.batchsize_eval)
shard_matrix, shard_strategy, unique_table_ids, reduction_table_ids = sharding.generate_plan(
    args.TABLE_SIZE_ARRAY,
    args.MULTI_HOT_SIZES,
    args.EMB_VEC_SIZES,
    num_nodes,
    args.num_gpus_per_node,
    args,
    is_rank_zero,
)
compression_strategy = {
    hugectr.CompressionStrategy.Unique: unique_table_ids,
    hugectr.CompressionStrategy.Reduction: reduction_table_ids,
}
# 1. Create Solver, DataReaderParams and Optimizer
solver = hugectr.CreateSolver(
    model_name="Embedding Collection Benchmark",
    max_eval_batches=args.max_eval_batches,
    batchsize_eval=args.batchsize_eval,
    batchsize=args.batchsize,
    vvgpu=[[x for x in range(args.num_gpus_per_node)] for _ in range(num_nodes)],
    repeat_dataset=True,
    i64_input_key=args.i64_input_key,
    lr=args.lr,
    warmup_steps=args.warmup_steps,
    decay_start=args.decay_start,
    decay_steps=args.decay_steps,
    decay_power=2.0,
    end_lr=0.0,
    use_mixed_precision=args.use_mixed_precision,
    enable_tf32_compute=args.enable_tf32_compute,
    scaler=args.scaler,
    use_cuda_graph=args.disable_cuda_graph,
    gen_loss_summary=args.gen_loss_summary,
    train_intra_iteration_overlap=args.disable_train_intra_iteration_overlap,
    train_inter_iteration_overlap=args.disable_train_inter_iteration_overlap,
    eval_intra_iteration_overlap=False,
    eval_inter_iteration_overlap=True,
    all_reduce_algo=hugectr.AllReduceAlgo.NCCL,
    grouped_all_reduce=True,
    num_iterations_statistics=20,
    metrics_spec={hugectr.MetricsType.AUC: args.auc_threshold},
    perf_logging=args.perf_logging,
    drop_incomplete_batch=True,
    use_embedding_collection=True,
    use_algorithm_search=args.use_algorithm_search,
)

optimizer = None
if args.optimizer == "adagrad":
    optimizer = hugectr.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.AdaGrad,
        update_type=hugectr.Update_t.Global,
        initial_accu_value=args.init_accu,
        epsilon=args.eps,
    )
elif args.optimizer == "sgd":
    optimizer = hugectr.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.SGD,
        update_type=hugectr.Update_t.Local,
        atomic_update=True,
    )

reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync,
    source=[args.dataset_path],
    eval_source=args.dataset_path,
    check_type=hugectr.Check_t.Non,
    num_samples=args.train_num_samples,
    eval_num_samples=args.eval_num_samples,
    cache_eval_data=1,
    slot_size_array=args.TABLE_SIZE_ARRAY,
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
        dense_dim=args.dense_dim,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("data{}".format(i), args.MULTI_HOT_SIZES[i], True, 1)
            for i in range(args.NUM_TABLE)
        ],
    )
)

# create embedding table
embedding_table_list = []
for i in range(args.NUM_TABLE):
    embedding_table_list.append(
        hugectr.EmbeddingTableConfig(
            name=str(i), max_vocabulary_size=args.TABLE_SIZE_ARRAY[i], ev_size=args.EMB_VEC_SIZES[i]
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
    table_config=[embedding_table_list[i] for i in range(args.NUM_TABLE)],
    bottom_name=["data{}".format(i) for i in range(args.NUM_TABLE)],
    top_name="sparse_embedding",
    combiner=args.COMBINERS,
)

ebc_config.shard(
    shard_matrix=shard_matrix,
    shard_strategy=shard_strategy,
    compression_strategy=compression_strategy,
)

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
        num_outputs=args.bmlp_layer_sizes,
        act_type=hugectr.Activation_t.Relu,
        compute_config=compute_config,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Concat,
        bottom_names=["sparse_embedding", "mlp1"],
        top_names=["concat1"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MultiCross,
        bottom_names=["concat1"],
        top_names=["interaction1"],
        projection_dim=args.dcn_low_rank_dim,
        num_layers=args.dcn_num_layers,
        compute_config=compute_config,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MLP,
        bottom_names=["interaction1"],
        top_names=["mlp2"],
        num_outputs=args.tmlp_layer_sizes,
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

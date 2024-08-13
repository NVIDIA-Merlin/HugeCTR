#!/usr/bin/env python3
# # SPDX-FileCopyrightText: Copyright (c) <2024> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
from typing import List, Optional

import torch
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.utils.data import IterableDataset
from data.multi_hot_criteo import CriteoRecDataset
from torchrec.datasets.random import RandomRecDataset
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs_registry,
    QCommsConfig,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingType,
)
from torchrec.distributed.types import (
    BoundsCheckMode,
    ShardingEnv,
)
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    DefaultDataParallelWrapper,
)
from dlrm import DLRM_DCN_Train
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from tqdm import tqdm
import sys
from fbgemm_gpu.split_table_batched_embeddings_ops_training import SparseType


def _get_random_dataset(
    cat_names: List[int],
    hash_sizes: List[int],
    ids_per_features: List[int],
    num_dense: int,
    batch_size: int = 32,
) -> IterableDataset:
    return RandomRecDataset(
        keys=cat_names,
        batch_size=batch_size,
        hash_sizes=hash_sizes,
        ids_per_features=ids_per_features,
        num_dense=num_dense,
        min_ids_per_feature=0,
        num_generated_batches=1,
    )


def table_idx_to_name(i):
    return f"t_{i}"


def feature_idx_to_name(i):
    return f"cate_{i}"


def get_comm_precission(precision_str):
    if precision_str == "fp32":
        return CommType.FP32
    elif precision_str == "fp16":
        return CommType.FP16
    elif precision_str == "bf16":
        return CommType.BF16
    elif precision_str == "fp8":
        return CommType.FP8
    else:
        raise ValueError("unknown comm precision type")


def get_planner(args, device):
    dict_const = {}
    for i in range(args.num_embedding_table):
        if (
            args.data_parallel_embeddings is not None
            and i in args.data_parallel_embeddings
        ):
            const = ParameterConstraints(
                sharding_types=[ShardingType.DATA_PARALLEL.value],
                # min_partition=2,
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
            )
        else:
            const = ParameterConstraints(
                sharding_types=[
                    # ShardingType.TABLE_WISE.value,
                    # ShardingType.COLUMN_WISE.value,
                    # ShardingType.ROW_WISE.value,
                    ShardingType.TABLE_ROW_WISE.value,
                    # ShardingType.TABLE_COLUMN_WISE.value,
                ],
                # min_partition=2,
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
            )
        dict_const[table_idx_to_name(i)] = const
    return EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
            hbm_cap=args.hbm_cap,
            intra_host_bw=args.intra_host_bw,
            inter_host_bw=args.inter_host_bw,
        ),
        constraints=dict_const,
        batch_size=args.batch_size,
        # # If experience OOM, increase the percentage. see
        # # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )


def train(args) -> None:
    """
    Constructs and trains a DLRM model (using random dummy data). Each script is run on each process (rank) in SPMD fashion.
    The embedding layers will be sharded across available ranks

    qcomm_forward_precision: Compression used in forwards pass. FP16 is the recommended usage. INT8 and FP8 are in development, but feel free to try them out.
    qcomm_backward_precision: Compression used in backwards pass. We recommend using BF16 to ensure training stability.

    The effects of quantized comms will be most apparent in large training jobs across multiple nodes where inter host communication is expensive.
    """
    # Init process_group , device, rank, backend
    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    dist.init_process_group(backend=backend)

    def all2all_warmup():
        dummy_input = (
            torch.arange(dist.get_world_size(), device="cuda")
            + dist.get_rank() * dist.get_world_size()
        )
        dummy_output = torch.empty(
            dist.get_world_size(), dtype=torch.int64, device="cuda"
        )
        dist.all_to_all_single(dummy_output, dummy_input)
        torch.cuda.synchronize()
        if rank == 0:
            print("all to all warmup finish", flush=True)

    all2all_warmup()
    args.batch_size = int(args.batch_size / dist.get_world_size())
        if rank == 0:
            print("batchsize_per_gpu", args.batch_size, flush=True)


    # Construct DLRM module
    eb_configs = [
        EmbeddingBagConfig(
            name=table_idx_to_name(feature_idx),
            embedding_dim=args.embedding_dims[feature_idx],
            num_embeddings=args.num_embeddings_per_feature[feature_idx],
            feature_names=[feature_idx_to_name(feature_idx)],
        )
        for feature_idx in range(args.num_embedding_table)
    ]
    train_model = DLRM_DCN_Train(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        dense_in_features=args.dense_in_features,
        dense_arch_layer_sizes=args.dense_arch_layer_sizes,
        over_arch_layer_sizes=args.over_arch_layer_sizes,
        dcn_num_layers=args.dcn_num_layers,
        dcn_low_rank_dim=args.dcn_low_rank_dim,
        batch_size=args.batch_size,
        bmlp_overlap=args.bmlp_overlap,
        enable_cuda_graph=args.enable_cuda_graph,
        skip_embedding=False,
        dense_device=device,
    )
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.optimizer_type == "sgd":
        embedding_optimizer = torch.optim.SGD
    elif args.optimizer_type == "adagrad":
        embedding_optimizer = torch.optim.Adagrad
        optimizer_kwargs["eps"] = args.eps
    elif args.optimizer_type == "rowwise_adagrad":
        embedding_optimizer = torch.optim.RowWiseAdagrad
    else:
        raise ValueError("unknown optimizer type")

    planner = get_planner(args, device)

    sharder = EmbeddingBagCollectionSharder(
        fused_params={
            "output_dtype": SparseType.FP16,
        }
    )

    def optimizer_with_params():
        if args.optimizer_type == "sgd":
            return lambda params: torch.optim.SGD(params, lr=args.learning_rate)
        elif args.optimizer_type == "adagrad":
            return lambda params: torch.optim.Adagrad(
                params, lr=args.learning_rate, eps=args.eps
            )
        elif args.optimizer_type == "rowwise_adagrad":
            return lambda params: torch.optim.Adagrad(
                params, lr=args.learning_rate, eps=args.eps
            )
        else:
            raise ValueError("unknown optimizer type")

    side_stream = torch.cuda.Stream()

    def model_parallel_shard(train_model):
        plan = planner.collective_plan(train_model, [sharder], dist.GroupMember.WORLD)

        if hasattr(train_model, "model"):
            apply_optimizer_in_backward(
                embedding_optimizer,
                train_model.model.sparse_arch.parameters(),
                optimizer_kwargs,
            )
        else:
            apply_optimizer_in_backward(
                embedding_optimizer,
                train_model.sparse_arch.parameters(),
                optimizer_kwargs,
            )

        with torch.cuda.stream(side_stream):
            model = DistributedModelParallel(
                module=train_model,
                device=device,
                # pyre-ignore
                sharders=[sharder],
                plan=plan,
            )

            non_fused_optimizer = KeyedOptimizerWrapper(
                dict(in_backward_optimizer_filter(model.named_parameters())),
                optimizer_with_params(),
            )
        torch.cuda.current_stream().wait_stream(side_stream)
        return model, non_fused_optimizer

    model, non_fused_optimizer = model_parallel_shard(train_model)

    if rank == 0 and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for table_name, plan in plans.items():
                print(table_name, "\n", plan, "\n")
    # Overlap comm/compute/device transfer during training through train_pipeline
    train_pipeline = TrainPipelineSparseDist(
        model,
        non_fused_optimizer,
        device,
    )

    # train model
    if args.use_criteo_multi_hot_data:
        train_iterator = iter(
            CriteoRecDataset(
                dataset_path=args.dataset_path,
                batch_size_per_gpu=args.batch_size,
                keys=[
                    feature_idx_to_name(feature_idx)
                    for feature_idx in range(args.num_embedding_table)
                ],
                num_dense=args.dense_in_features,
                ids_per_features=args.multi_hot_sizes,
                rank=dist.get_rank(),
            )
        )
    else:
        train_iterator = iter(
            _get_random_dataset(
                cat_names=[
                    feature_idx_to_name(feature_idx)
                    for feature_idx in range(args.num_embedding_table)
                ],
                hash_sizes=args.num_embeddings_per_feature,
                ids_per_features=args.multi_hot_sizes,
                num_dense=args.dense_in_features,
                batch_size=args.batch_size,
            )
        )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in tqdm(range(int(args.num_iterations) + 210), mininterval=5.0):
        if i == 200:
            torch.cuda.cudart().cudaProfilerStart()
            start.record()
            train_pipeline.cpu_launch_time = 0
        if i == args.num_iterations + 200:
            end.record()
            torch.cuda.cudart().cudaProfilerStop()

        train_pipeline.progress(train_iterator)

    torch.cuda.synchronize()
    if rank == 0:
        print(
            "time per iteration:{:.5}ms".format(
                start.elapsed_time(end) / (args.num_iterations),
            )
        )


@record
def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="number of iterations",
    )
    parser.add_argument(
        "--train_num_samples",
        type=int,
        default=100,
        help="number of iterations",
    )
    parser.add_argument(
        "--eval_num_samples",
        type=int,
        default=100,
        help="number of iterations",
    )
    parser.add_argument(
        "--num_table",
        help="numuber of each type of embedding table, can be an integer or list of integer seperated by comma",
        type=str,
        default="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
    )
    parser.add_argument(
        "--vocabulary_size_per_table",
        help="numuber of vocabulary_size of each type of embedding table, can be a list of integer seperated by comma",
        type=str,
        default="40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36",
    )
    parser.add_argument(
        "--nnz_per_table",
        help="numuber of nnz of each type of embedding table, can be a list of integer seperated by comma",
        type=str,
        default="3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1",
    )
    parser.add_argument(
        "--ev_size_per_table",
        help="ev_size of each type of embedding table, can be an integer or a list of integer seperated by comma",
        type=str,
        default="128",
    )

    # parser.add_argument(
    #     "--num_embeddings_per_feature",
    #     type=str,
    #     default="100,200",
    #     help="Comma separated max_ind_size per sparse feature. The number of embeddings"
    #     " in each embedding table. 26 values are expected for the Criteo dataset.",
    # )
    # parser.add_argument(
    #     "--multi_hot_sizes",
    #     type=str,
    #     default="1,100",
    #     help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    # )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
    )
    parser.add_argument(
        "--fwd_a2a_precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--bck_a2a_precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--allreduce_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
    )
    # parser.add_argument(
    #     "--embedding_dim",
    #     type=int,
    #     default=128,
    #     help="Size of each embedding.",
    # )
    parser.add_argument(
        "--dense_dim",
        type=int,
        default=13,
        help="dense_dim.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,128",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="1024,1024,512,256,1",
        help="Comma separated layer sizes for over arch.",
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
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adagrad",
        choices=["sgd", "adagrad", "rowwise_adagrad"],
        help="optimzier type.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--data_parallel_embeddings",
        type=str,
        default=None,
        help="Comma separated data parallell embedding table ids.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="a100",
        choices=["a100", "h100", "h200"],
        help="Platform, has different system spec",
    )
    parser.add_argument(
        "--use_criteo_multi_hot_data",
        action="store_true",
        help="use criteo multi hot data",
    )
    parser.add_argument(
        "--bmlp_overlap",
        action="store_true",
        help="overlap bottom mlp",
    )
    parser.add_argument(
        "--enable_cuda_graph",
        action="store_true",
        help="enable cuda_graph",
    )
    args = parser.parse_args()

    args.batch_size = args.batchsize
    args.optimizer_type = args.optimizer
    args.dense_in_features = args.dense_dim

    vocabulary_size_per_table = args.vocabulary_size_per_table.split(",")
    nnz_per_table = args.nnz_per_table.split(",")
    ev_size_per_table = args.ev_size_per_table.split(",")
    args.num_embeddings_per_feature = []
    args.multi_hot_sizes = []
    args.embedding_dims = []
    for i, num_table in enumerate([int(v) for v in args.num_table.split(",")]):
        for table_id in range(num_table):
            args.num_embeddings_per_feature.append(int(vocabulary_size_per_table[i]))
            args.multi_hot_sizes.append(int(nnz_per_table[i]))
            args.embedding_dims.append(int(ev_size_per_table[i]))

    args.dense_arch_layer_sizes = [
        int(v) for v in args.dense_arch_layer_sizes.split(",")
    ]
    args.over_arch_layer_sizes = [int(v) for v in args.over_arch_layer_sizes.split(",")]
    args.data_parallel_embeddings = (
        None
        if args.data_parallel_embeddings is None
        else [int(v) for v in args.data_parallel_embeddings.split(",")]
    )

    args.num_embedding_table = len(args.num_embeddings_per_feature)

    if args.platform == "a100":
        args.intra_host_bw = 300e9
        args.inter_host_bw = 25e9
        args.hbm_cap = 80 * 1024 * 1024 * 1024
    elif args.platform == "h100":
        args.intra_host_bw = 450e9
        args.inter_host_bw = 25e9  # TODO: need check
        args.hbm_cap = 80 * 1024 * 1024 * 1024
    elif args.platform == "h200":
        args.intra_host_bw = 450e9
        args.inter_host_bw = 450e9
        args.hbm_cap = 140 * 1024 * 1024 * 1024

    train(args)


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import sys
from typing import Iterator, List, Optional, cast

import torch
from torch import distributed as dist, nn
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection, EmbeddingCollection
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner.proposers import (
    GreedyProposer,
    GridSearchProposer,
    UniformProposer,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology, ParameterConstraints
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fused_embeddingbag import FusedEmbeddingBagCollectionSharder
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingType,
)
from torchrec.distributed.types import (
    BoundsCheckMode,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from tqdm import tqdm

from data.dlrm_dataloader import get_dataloader  # noqa F811
from lr_scheduler import LRPolicyScheduler  # noqa F811
from dlrm import DLRM_DCN, DLRMTrain

# from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
from itertools import product

TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.

import ctypes

_cudart = ctypes.CDLL("libcudart.so")


def post_process_args(args):
    num_table_list = args.num_table.strip().split(",")
    vocabulary_size_per_table = args.vocabulary_size_per_table.strip().split(",")
    nnz_per_table = args.nnz_per_table.split(",")
    ev_size_per_table = args.ev_size_per_table.split(",")
    combiner_per_table = args.combiner_per_table.split(",")
    num_vocabulary_size = len(vocabulary_size_per_table)

    assert len(num_table_list) == num_vocabulary_size
    assert len(nnz_per_table) == num_vocabulary_size
    assert len(ev_size_per_table) == 1 or len(ev_size_per_table) == num_vocabulary_size
    assert len(combiner_per_table) == num_vocabulary_size
    if len(ev_size_per_table) == 1:
        ev_size_per_table = [ev_size_per_table[0] for _ in range(num_vocabulary_size)]
    num_table_list = [int(v) for v in num_table_list]

    TABLE_SIZE_ARRAY = []
    MULTI_HOT_SIZES = []
    COMBINERS = []
    EMB_VEC_SIZES = []
    for i, num_table in enumerate(num_table_list):
        for _ in range(num_table):
            TABLE_SIZE_ARRAY.append(int(vocabulary_size_per_table[i]))
            MULTI_HOT_SIZES.append(int(nnz_per_table[i]))
            EMB_VEC_SIZES.append(int(ev_size_per_table[i]))
            combiner = combiner_per_table[i]
            if combiner == "s":
                COMBINERS.append("sum")
            elif combiner == "c":
                COMBINERS.append("concat")
            else:
                raise

    NUM_TABLE = len(TABLE_SIZE_ARRAY)

    args.NUM_TABLE = NUM_TABLE
    args.TABLE_SIZE_ARRAY = TABLE_SIZE_ARRAY
    args.MULTI_HOT_SIZES = MULTI_HOT_SIZES
    args.COMBINERS = COMBINERS
    args.EMB_VEC_SIZES = EMB_VEC_SIZES

    args.limit_train_batches = args.train_num_samples // args.batchsize
    args.dense_arch_layer_sizes = args.bmlp_layer_sizes
    args.over_arch_layer_sizes = args.tmlp_layer_sizes
    return args


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--train_num_samples",
        type=int,
        default=None,
        help="number of train batches",
    )
    parser.add_argument(
        "--eval_num_samples",
        type=int,
        default=None,
        help="number of train batches",
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
    parser.add_argument(
        "--combiner_per_table",
        help="combiner of each type of embedding table, can be a list of str seperated by comma. str can be c(concat) or s(sum)",
        type=str,
        default="s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s",
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
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--tmlp_layer_sizes",
        type=str,
        default="512,512,256,1",
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
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="dataset path",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_decay_start", type=int, default=0)
    parser.add_argument("--lr_decay_steps", type=int, default=0)
    parser.add_argument(
        "--print_lr",
        action="store_true",
        help="Print learning rate every iteration.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon for Adagrad optimizer.",
    )
    parser.set_defaults(
        pin_memory=None,
        mmap_mode=None,
        drop_last=None,
        shuffle_batches=None,
        shuffle_training_set=None,
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
    )

    return post_process_args(parser.parse_args(argv))


def batched(it: Iterator, n: int):
    assert n >= 1
    for x in it:
        yield itertools.chain((x,), itertools.islice(it, n - 1))


def _train(
    pipeline: TrainPipelineSparseDist,
    train_dataloader: DataLoader,
    epoch: int,
    lr_scheduler,
    print_lr: bool,
    limit_train_batches: Optional[int],
) -> None:
    """
    Trains model for 1 epoch. Helper function for train_val_test.

    Args:
        pipeline (TrainPipelineSparseDist): data pipeline.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        epoch (int): The number of complete passes through the training set so far.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.
        print_lr (bool): Whether to print the learning rate every training step.
        validation_freq (Optional[int]): The number of training steps between validation runs within an epoch.
        limit_train_batches (Optional[int]): Limits the training set to the first `limit_train_batches` batches.
        limit_val_batches (Optional[int]): Limits the validation set to the first `limit_val_batches` batches.

    Returns:
        None.
    """
    pipeline._model.train()
    n = limit_train_batches if limit_train_batches else len(train_dataloader)
    iterator = itertools.islice(iter(train_dataloader), n)

    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Epoch {epoch}",
            total=n,
            disable=False,
        )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    i = 0
    for b in batched(iterator, 1):
        if i == 20:
            _cudart.cudaProfilerStart()
            start.record()

        pipeline.progress(b)
        lr_scheduler.step()
        if is_rank_zero:
            pbar.update(1)
        i += 1
    _cudart.cudaProfilerStop()
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    if is_rank_zero:
        print(
            f"num_iteration:{i-20}, time:{elapsed_time}ms, time per iter:{elapsed_time / (i - 20)}"
        )


def train(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_dataloader: DataLoader,
    lr_scheduler: LRPolicyScheduler,
) -> None:
    """
    Train/validation/test loop.

    Args:
        args (argparse.Namespace): parsed command line args.
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device to use.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        test_dataloader (DataLoader): Test set's dataloader.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.

    Returns:
        TrainValTestResults.
    """
    pipeline = TrainPipelineSparseDist(model, optimizer, device, execute_all_batches=True)

    for epoch in range(args.epochs):
        _train(
            pipeline,
            train_dataloader,
            epoch,
            lr_scheduler,
            args.print_lr,
            args.limit_train_batches,
        )


def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """
    args = parse_args(argv)
    for name, val in vars(args).items():
        try:
            vars(args)[name] = list(map(int, val.split(",")))
        except (ValueError, AttributeError):
            pass

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = int(dist.get_world_size())
    args.batch_size_per_gpu = args.batchsize // world_size

    if rank == 0:
        print(
            "PARAMS: (lr, batch_size_per_gpu, warmup_steps, decay_start, decay_steps): "
            f"{(args.learning_rate, args.batch_size_per_gpu, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps)}"
        )

    train_dataloader = get_dataloader(args, backend, "train")

    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = args.over_arch_layer_sizes

    def group_embedding_with_same_ev_size():
        ebc = []
        unique_ev_sizes = sorted(list(set(args.EMB_VEC_SIZES)))
        unique_combiners = sorted(list(set(args.COMBINERS)))
        for ev_size, combiner in product(unique_ev_sizes, unique_combiners):
            filtered_table_ids = []
            for i in range(args.NUM_TABLE):
                if ev_size != args.EMB_VEC_SIZES[i] or combiner != args.COMBINERS[i]:
                    continue
                filtered_table_ids.append(i)
            if len(filtered_table_ids) == 0:
                continue
            embedding = EmbeddingBagConfig if combiner == "sum" else EmbeddingConfig
            embedding_collection = (
                EmbeddingBagCollection if combiner == "sum" else EmbeddingCollection
            )
            eb_configs = [
                embedding(
                    name=f"t_table_{feature_idx}",
                    embedding_dim=args.EMB_VEC_SIZES[feature_idx],
                    num_embeddings=args.TABLE_SIZE_ARRAY[feature_idx],
                    feature_names=["table{}".format(feature_idx)],
                )
                for feature_idx in filtered_table_ids
            ]
            ebc.append(embedding_collection(tables=eb_configs, device=torch.device("meta")))
        return ebc

    def group_embedding_using_embedding_bag_collection():
        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_table_{feature_idx}",
                embedding_dim=args.EMB_VEC_SIZES[feature_idx],
                num_embeddings=args.TABLE_SIZE_ARRAY[feature_idx],
                feature_names=["table{}".format(feature_idx)],
            )
            for feature_idx in range(args.NUM_TABLE)
        ]
        ebc = [EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))]
        return ebc

    # def generate_embedding_collection_with_same_ev_size():
    #     eb_configs = [
    #         EmbeddingBagConfig(
    #             name=f"t_table_{feature_idx}",
    #             embedding_dim=args.EMB_VEC_SIZES[feature_idx],
    #             num_embeddings=args.TABLE_SIZE_ARRAY[feature_idx],
    #             feature_names = ["table{}".format(feature_idx)],
    #         )
    #         for feature_idx in range(args.NUM_TABLE)
    #     ]
    #     return EmbeddingBagCollection(
    #         tables=eb_configs, device=torch.device("meta")
    #     )

    dlrm_model = DLRM_DCN(
        # embedding_bag_collection=generate_embedding_collection_with_same_ev_size(),
        embedding_bag_collection=group_embedding_using_embedding_bag_collection(),
        dense_in_features=args.dense_dim,
        dense_arch_layer_sizes=args.dense_arch_layer_sizes,
        over_arch_layer_sizes=args.over_arch_layer_sizes,
        dcn_num_layers=args.dcn_num_layers,
        dcn_low_rank_dim=args.dcn_low_rank_dim,
        dense_device=device,
    )

    train_model = DLRMTrain(dlrm_model)
    embedding_optimizer = torch.optim.Adagrad if args.adagrad else torch.optim.SGD
    # This will apply the Adagrad optimizer in the backward pass for the embeddings (sparse_arch). This means that
    # the optimizer update will be applied in the backward pass, in this case through a fused op.
    # TorchRec will use the FBGEMM implementation of EXACT_ADAGRAD. For GPU devices, a fused CUDA kernel is invoked. For CPU, FBGEMM_GPU invokes CPU kernels
    # https://github.com/pytorch/FBGEMM/blob/2cb8b0dff3e67f9a009c4299defbd6b99cc12b8f/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py#L676-L678

    # Note that lr_decay, weight_decay and initial_accumulator_value for Adagrad optimizer in FBGEMM v0.3.2
    # cannot be specified below. This equivalently means that all these parameters are hardcoded to zero.
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.adagrad:
        optimizer_kwargs["eps"] = args.eps
    apply_optimizer_in_backward(
        embedding_optimizer,
        train_model.model.sparse_arch.parameters(),
        optimizer_kwargs,
    )
    dict_const = {}
    for i in range(args.NUM_TABLE):
        const = ParameterConstraints(
            # sharding_types=[
            #     ShardingType.TABLE_WISE.value,
            #     ShardingType.COLUMN_WISE.value,
            #     ShardingType.ROW_WISE.value,
            #     ShardingType.TABLE_ROW_WISE.value,
            #     ShardingType.TABLE_COLUMN_WISE.value,
            # ],
            # min_partition=2,
            pooling_factors=[args.MULTI_HOT_SIZES[i]],
            num_poolings=[1],
            enforce_hbm=True,
            bounds_check_mode=BoundsCheckMode.NONE,
        )
        dict_const["t_table_" + str(i)] = const

    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
            hbm_cap=80 * 1024 * 1024 * 1024,
            intra_host_bw=300e9,
            inter_host_bw=25e9,
        ),
        constraints=dict_const,
        batch_size=args.batch_size_per_gpu,
        # # If experience OOM, increase the percentage. see
        # # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )
    plan = planner.collective_plan(train_model, get_default_sharders(), dist.GroupMember.WORLD)

    model = DistributedModelParallel(
        module=train_model,
        device=device,
        plan=plan,
    )
    if rank == 0 and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for table_name, plan in plans.items():
                print(table_name, "\n", plan, "\n")

    def optimizer_with_params():
        if args.adagrad:
            return lambda params: torch.optim.Adagrad(params, lr=args.learning_rate, eps=args.eps)
        else:
            return lambda params: torch.optim.SGD(params, lr=args.learning_rate)

    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        optimizer_with_params(),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    lr_scheduler = LRPolicyScheduler(
        optimizer, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps
    )

    train(
        args,
        model,
        optimizer,
        device,
        train_dataloader,
        lr_scheduler,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

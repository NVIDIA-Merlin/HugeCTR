#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import List

from torch import distributed as dist
from torch.utils.data import DataLoader

from .multi_hot_criteo import RawDataIterDataPipe  # noqa F811

STAGES = ["train", "val"]


def _get_in_memory_dataloader(
    args: argparse.Namespace,
    stage: str,
) -> DataLoader:
    if stage in ["val"] and args.test_batch_size is not None:
        batch_size_per_gpu = args.test_batch_size
    else:
        batch_size_per_gpu = args.batch_size_per_gpu
    dataloader = DataLoader(
        RawDataIterDataPipe(
            args.dataset_path,
            batch_size_per_gpu=batch_size_per_gpu,
            dense_dim=args.dense_dim,
            sparse_dim=args.MULTI_HOT_SIZES,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        ),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader


def get_dataloader(args: argparse.Namespace, backend: str, stage: str) -> DataLoader:
    """
    Gets desired dataloader from dlrm_main command line options. Currently, this
    function is able to return either a DataLoader wrapped around a RandomRecDataset or
    a Dataloader wrapped around an InMemoryBinaryCriteoIterDataPipe.

    Args:
        args (argparse.Namespace): Command line options supplied to dlrm_main.py's main
            function.
        backend (str): "nccl" or "gloo".
        stage (str): "train", "val", or "test".

    Returns:
        dataloader (DataLoader): PyTorch dataloader for the specified options.

    """
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    args.pin_memory = (backend == "nccl") if not hasattr(args, "pin_memory") else args.pin_memory

    return _get_in_memory_dataloader(args, stage)

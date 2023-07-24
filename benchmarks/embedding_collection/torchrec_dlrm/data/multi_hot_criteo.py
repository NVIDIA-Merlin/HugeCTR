#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import zipfile
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

import struct
import os


class RawDataIterDataPipe(IterableDataset):
    def __init__(
        self,
        dataset_path: str,
        batch_size_per_gpu: int,
        dense_dim: int,
        sparse_dim: List[int],
        rank: int,
        world_size: int,
    ) -> None:
        self.dataset_path = dataset_path
        self.batch_size_per_gpu = batch_size_per_gpu
        self.dense_dim = dense_dim
        self.sparse_dim = sparse_dim
        self.rank = rank
        self.world_size = world_size

        self.num_cate_features = sum(sparse_dim)
        self.item_num_per_sample = 1 + dense_dim + self.num_cate_features
        self.sample_format = r"1I" + str(dense_dim) + "f" + str(self.num_cate_features) + "I"
        self.sample_size_in_bytes = self.item_num_per_sample * 4
        self.file_size = os.path.getsize(dataset_path)
        self.num_full_batches = self.file_size // self.sample_size_in_bytes
        self.num_table = len(sparse_dim)
        self.keys = ["table{}".format(i) for i in range(self.num_table)]
        self.index_per_key: Dict[str, int] = {key: i for (i, key) in enumerate(self.keys)}
        self._init_cache()

    def _np_arrays_to_batch(
        self,
        dense: np.ndarray,
        sparse: List[np.ndarray],
        labels: np.ndarray,
    ) -> Batch:
        batch_size = self.batch_size_per_gpu
        lengths = torch.ones((self.num_table * batch_size), dtype=torch.int32)
        for k, multi_hot_size in enumerate(self.sparse_dim):
            lengths[k * batch_size : (k + 1) * batch_size] = multi_hot_size
        offsets = torch.cumsum(torch.concat((torch.tensor([0]), lengths)), dim=0)
        length_per_key = [batch_size * multi_hot_size for multi_hot_size in self.sparse_dim]
        offset_per_key = torch.cumsum(
            torch.concat((torch.tensor([0]), torch.tensor(length_per_key))), dim=0
        )
        values = torch.concat([torch.from_numpy(feat).flatten() for feat in sparse])
        return Batch(
            dense_features=torch.from_numpy(dense.copy()),
            sparse_features=KeyedJaggedTensor(
                keys=self.keys,
                values=values,
                lengths=lengths,
                offsets=offsets,
                stride=batch_size,
                length_per_key=length_per_key,
                offset_per_key=offset_per_key.tolist(),
                index_per_key=self.index_per_key,
            ),
            labels=torch.from_numpy(labels.reshape(-1).copy()),
        ).to("cuda")

    def _init_cache(self):
        self.cache = []
        num_cache = 10

        batch_idx = 0
        with open(self.dataset_path, "rb") as file:
            while batch_idx < self.num_full_batches:
                if len(self.cache) >= num_cache:
                    break
                if batch_idx % self.world_size == self.rank:
                    offset = self.sample_size_in_bytes * batch_idx
                    file.seek(offset)
                    data_buffer = file.read(self.batch_size_per_gpu * self.sample_size_in_bytes)
                    data = struct.unpack(self.sample_format * self.batch_size_per_gpu, data_buffer)
                    data = [
                        list(
                            data[i * self.item_num_per_sample : (i + 1) * self.item_num_per_sample]
                        )
                        for i in range(self.batch_size_per_gpu)
                    ]
                    labels = np.asarray(
                        [data[i][0] for i in range(self.batch_size_per_gpu)]
                    ).astype(np.int32)
                    dense = np.asarray(
                        [data[i][1 : 1 + self.dense_dim] for i in range(self.batch_size_per_gpu)]
                    ).astype(np.float32)
                    sparse = []
                    for sparse_idx in range(len(self.sparse_dim)):
                        cumsum = sum(self.sparse_dim[:sparse_idx])
                        cur_sparse_dim = self.sparse_dim[sparse_idx]
                        sparse += [
                            np.asarray(
                                [
                                    data[i][
                                        1
                                        + self.dense_dim
                                        + cumsum : 1
                                        + self.dense_dim
                                        + cumsum
                                        + cur_sparse_dim
                                    ]
                                    for i in range(self.batch_size_per_gpu)
                                ]
                            ).astype(np.int32)
                        ]
                    self.cache.append(self._np_arrays_to_batch(dense, sparse, labels))
                else:
                    batch_idx += 1

    def __iter__(self) -> Iterator[Batch]:
        batch_idx = 0
        local_cache_idx = 0
        while batch_idx < self.num_full_batches:
            if batch_idx % self.world_size == self.rank:
                yield self.cache[local_cache_idx % len(self.cache)]
                local_cache_idx += 1
            batch_idx += 1

    def __len__(self) -> int:
        return self.num_full_batches // self.world_size

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import sys
from typing import cast, Iterator, List, Optional
import struct

import torch
from torch.utils.data.dataset import IterableDataset
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _CriteoRecBatch:
    def __init__(
        self,
        dataset_path: str,
        keys: List[str],
        batch_size: int,
        num_dense: int,
        ids_per_features: List[int],
        rank: int,
    ) -> None:
        self.dataset_path = dataset_path
        self.keys = keys
        self.keys_length: int = len(keys)
        self.batch_size = batch_size
        self.ids_per_features = ids_per_features
        self.num_dense = num_dense
        self.num_generated_batches = 100

        self._generated_batches: List[Batch] = [
            self._generate_batch(_ * rank) for _ in range(self.num_generated_batches)
        ]
        self.batch_index = 0
        self.num_batches = None

    def __iter__(self) -> "_CriteoRecBatch":
        self.batch_index = 0
        return self

    def __next__(self) -> Batch:
        if self.batch_index == self.num_batches:
            raise StopIteration
        if self.num_generated_batches >= 0:
            batch = self._generated_batches[
                self.batch_index % len(self._generated_batches)
            ]
        else:
            batch = self._generate_batch()
        self.batch_index += 1
        return batch

    def _generate_batch(self, batch_index) -> Batch:
        sample_format = (
            r"1I" + str(self.num_dense) + "f" + str(sum(self.ids_per_features)) + "I"
        )
        num_item_per_sample = 1 + self.num_dense + sum(self.ids_per_features)
        sample_size_in_bytes = num_item_per_sample * 4

        with open(self.dataset_path, "rb") as f:
            f.seek(sample_size_in_bytes * (batch_index * self.batch_size % 65536))
            values = [[] for _ in range(len(self.keys))]
            lengths = [[] for _ in range(len(self.keys))]
            dense_features = []
            labels = []
            for _ in range(self.batch_size):
                buffer = f.read(sample_size_in_bytes)
                data = struct.unpack(sample_format, buffer)
                labels.append(int(data[0]))
                for i in range(1, 1 + self.num_dense):
                    dense_features.append(float(data[i]))

                offset = 1 + self.num_dense
                for key_idx in range(len(self.keys)):
                    ids_in_current_feature = self.ids_per_features[key_idx]
                    lengths[key_idx].append(ids_in_current_feature)
                    for i in range(ids_in_current_feature):
                        values[key_idx].append(int(data[offset + i]))
                    offset += ids_in_current_feature

            values = list(itertools.chain.from_iterable(values))
            lengths = list(itertools.chain.from_iterable(lengths))

            sparse_features = KeyedJaggedTensor.from_lengths_sync(
                keys=self.keys,
                values=torch.tensor(values, dtype=torch.int64),
                lengths=torch.tensor(lengths, dtype=torch.int32),
            )
            dense_features = torch.tensor(dense_features, dtype=torch.float32).reshape(
                -1, self.num_dense
            )

            labels = torch.tensor(labels, dtype=torch.int64)

            batch = Batch(
                dense_features=dense_features,
                sparse_features=sparse_features,
                labels=labels,
            )
            return batch.pin_memory()


class CriteoRecDataset(IterableDataset[Batch]):
    """
    Random iterable dataset used to generate batches for recommender systems
    (RecSys). Currently produces unweighted sparse features only. TODO: Add
    weighted sparse features.

    Args:
        keys (List[str]): List of feature names for sparse features.
        batch_size (int): batch size.
        hash_size (Optional[int]): Max sparse id value. All sparse IDs will be taken
            modulo this value.
        hash_sizes (Optional[List[int]]): Max sparse id value per feature in keys. Each
            sparse ID will be taken modulo the corresponding value from this argument. Note, if this is used, hash_size will be ignored.
        ids_per_feature (int): Number of IDs per sparse feature.
        ids_per_features (int): Number of IDs per sparse feature in each key. Note, if this is used, ids_per_feature will be ignored.
        num_dense (int): Number of dense features.
        manual_seed (int): Seed for deterministic behavior.
        num_batches: (Optional[int]): Num batches to generate before raising StopIteration
        num_generated_batches int: Num batches to cache. If num_batches > num_generated batches, then we will cycle to the first generated batch.
                                   If this value is negative, batches will be generated on the fly.
        min_ids_per_feature (int): Minimum number of IDs per features.

    Example::

        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_size=100_000,
            ids_per_feature=1,
            num_dense=13,
        ),
        example = next(iter(dataset))
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size_per_gpu: int,
        keys,
        num_dense: int,
        ids_per_features: List[int],
        rank: int,
    ) -> None:
        super().__init__()

        assert len(ids_per_features) == len(
            keys
        ), "length of ids_per_features must be equal to the number of keys"

        self.batch_generator = _CriteoRecBatch(
            dataset_path=dataset_path,
            keys=keys,
            batch_size=batch_size_per_gpu,
            num_dense=num_dense,
            ids_per_features=ids_per_features,
            rank=rank,
        )
        self.num_batches: int = cast(int, sys.maxsize)

    def __iter__(self) -> Iterator[Batch]:
        return itertools.islice(iter(self.batch_generator), self.num_batches)

    def __len__(self) -> int:
        return self.num_batches

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.modules.crossnet import LowRankCrossNet
from torchrec.modules.embedding_modules import EmbeddingBagCollection, EmbeddingCollection
from torchrec.modules.mlp import MLP
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor, JaggedTensor


def choose(n: int, k: int) -> int:
    """
    Simple implementation of math.comb for Python 3.7 compatibility.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


# class SparseArch(nn.Module):
#     """
#     Processes the sparse features of DLRM. Does embedding lookups for all EmbeddingBag
#     and embedding features of each collection.

#     Args:
#         embedding_bag_collection (EmbeddingBagCollection): represents a collection of
#             pooled embeddings.

#     Example::

#         eb1_config = EmbeddingBagConfig(
#            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
#         )
#         eb2_config = EmbeddingBagConfig(
#            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
#         )

#         ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
#         sparse_arch = SparseArch(embedding_bag_collection)

#         #     0       1        2  <-- batch
#         # 0   [0,1] None    [2]
#         # 1   [3]    [4]    [5,6,7]
#         # ^
#         # feature
#         features = KeyedJaggedTensor.from_offsets_sync(
#            keys=["f1", "f2"],
#            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
#            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
#         )

#         sparse_embeddings = sparse_arch(features)
#     """

#     def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
#         super().__init__()
#         self.embedding_bag_collection: EmbeddingBagCollection = nn.ModuleList(embedding_bag_collection)
#         self._sparse_feature_names = []
#         for ebc in embedding_bag_collection:
#             conf = ebc.embedding_bag_configs() if isinstance(ebc, EmbeddingBagCollection) else ebc.embedding_configs()
#             self._sparse_feature_names.append([c.feature_names[0] for c in conf])

#     def forward(
#             self,
#             features: KeyedJaggedTensor,
#     ) -> torch.Tensor:
#         """
#         Args:
#             features (KeyedJaggedTensor): an input tensor of sparse features.

#         Returns:
#             torch.Tensor: tensor of shape B X F X D.
#         """
#         sparse_values: List[torch.Tensor] = []

#         for i, ebc in enumerate(self.embedding_bag_collection):

#             sparse_features = ebc(features)
#             for name in self._sparse_feature_names[i]:
#                 sparse_value = sparse_features[name]
#                 sparse_value = sparse_value.values() if  isinstance(sparse_value, JaggedTensor) else sparse_value
#                 sparse_values.append(sparse_value)
#         return torch.cat(sparse_values, dim=1)

#     @property
#     def sparse_feature_names(self) -> List[str]:
#         return self._sparse_feature_names


class SparseArch(nn.Module):
    """
    Processes the sparse features of DLRM. Does embedding lookups for all EmbeddingBag
    and embedding features of each collection.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): represents a collection of
            pooled embeddings.

    Example::

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(embedding_bag_collection)

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f2"],
           values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
           offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sparse_embeddings = sparse_arch(features)
    """

    def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
        super().__init__()
        assert len(embedding_bag_collection) == 1
        assert isinstance(embedding_bag_collection[0], EmbeddingBagCollection)
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection[0]
        self._sparse_feature_names = [
            c.feature_names[0] for c in self.embedding_bag_collection.embedding_bag_configs()
        ]

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """
        sparse_features: KeyedTensor = self.embedding_bag_collection(features)

        sparse: Dict[str, torch.Tensor] = sparse_features.to_dict()
        sparse_values: List[torch.Tensor] = []
        for name in self.sparse_feature_names:
            sparse_values.append(sparse[name])

        return torch.cat(sparse_values, dim=1)

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names


class DenseArch(nn.Module):
    """
    Processes the dense features of DLRM model.

    Args:
        in_features (int): dimensionality of the dense input features.
        layer_sizes (List[int]): list of layer sizes.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        dense_arch = DenseArch(10, layer_sizes=[15, D])
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(
            in_features, layer_sizes, bias=True, activation="relu", device=device
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): an input tensor of dense features.

        Returns:
            torch.Tensor: an output tensor of size B X D.
        """
        return self.model(features)


class InteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the pairwise dot product of each sparse feature pair,
    the dot product of each sparse features with the output of the dense layer,
    and the dense layer itself (all concatenated).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=len(keys))

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (D + F + F choose 2)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, num_sparse_features: int) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.triu_indices: torch.Tensor = torch.triu_indices(self.F + 1, self.F + 1, offset=1)

    def forward(self, dense_features: torch.Tensor, sparse_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (D + F + F choose 2).
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        combined_values = torch.cat((dense_features.unsqueeze(1), sparse_features), dim=1)
        # dense/sparse + sparse/sparse interaction
        # size B X (F + F choose 2)
        interactions = torch.bmm(combined_values, torch.transpose(combined_values, 1, 2))
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]

        return torch.cat((dense_features, interactions_flat), dim=1)


class InteractionDCNArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the output of a Deep Cross Net v2
    https://arxiv.org/pdf/2008.13535.pdf with a low rank approximation for the
    weight matrix. The input and output sizes are the same for this
    interaction layer (F*D + D).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        DCN = LowRankCrossNet(
            in_features = F*D+D,
            dcn_num_layers = 2,
            dnc_low_rank_dim = 4,
        )
        inter_arch = InteractionDCNArch(
            num_sparse_features=len(keys),
            crossnet=DCN,
        )

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (F*D + D)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, crossnet: nn.Module) -> None:
        super().__init__()
        self.crossnet = crossnet

    def forward(self, dense_features: torch.Tensor, sparse_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (F*D + D).
        """
        (B, D) = dense_features.shape

        combined_values = torch.cat((dense_features, sparse_features), dim=1)

        # size B X (F*D + D)
        return self.crossnet(combined_values.reshape([B, -1]))


class OverArch(nn.Module):
    """
    Final Arch of DLRM - simple MLP over OverArch.

    Args:
        in_features (int): size of the input.
        layer_sizes (List[int]): sizes of the layers of the `OverArch`.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        over_arch = OverArch(10, [5, 1])
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError("OverArch must have multiple layers.")
        self.model: nn.Module = nn.Sequential(
            MLP(
                in_features,
                layer_sizes[:-1],
                bias=True,
                activation="relu",
                device=device,
            ),
            nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True, device=device),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor):

        Returns:
            torch.Tensor: size B X layer_sizes[-1]
        """
        return self.model(features)


class DLRM_DCN(nn.Module):
    """
    Recsys model with DCN modified from the original model from "Deep Learning Recommendation
    Model for Personalization and Recommendation Systems"
    (https://arxiv.org/abs/1906.00091). Similar to DLRM module but has
    DeepCrossNet https://arxiv.org/pdf/2008.13535.pdf as the interaction layer.

    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually
            specified here.
        dcn_num_layers (int): the number of DCN layers in the interaction.
        dcn_low_rank_dim (int): the dimensionality of low rank approximation
            used in the dcn layers.
        dense_device (Optional[torch.device]): default compute device.

    Example::

        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2",
           embedding_dim=D,
           num_embeddings=100,
           feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM_DCN(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20, D],
           dcn_num_layers=2,
           dcn_low_rank_dim=8,
           over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f3"],
           values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
           offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
           dense_features=features,
           sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        embedding_bag_collection: List[EmbeddingBagCollection],
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dcn_num_layers: int,
        dcn_low_rank_dim: int,
        dense_device: Optional[torch.device] = None,
    ) -> None:
        # initialize DLRM
        # sparse arch and dense arch are initialized via DLRM
        super().__init__()
        assert len(embedding_bag_collection) > 0, "At least one embedding bag is required"
        embedding_dim_per_sample = 0
        for ebc in embedding_bag_collection:
            if isinstance(ebc, EmbeddingBagCollection):
                for ebc_config in ebc.embedding_bag_configs():
                    embedding_dim_per_sample += ebc_config.embedding_dim
            elif isinstance(ebc, EmbeddingCollection):
                for ebc_config in ebc.embedding_configs():
                    embedding_dim_per_sample += ebc_config.embedding_dim

        self.sparse_arch: SparseArch = SparseArch(embedding_bag_collection)

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=dense_device,
        )

        # Fix interaction and over arch for DLRM_DCN
        over_in_features: int = embedding_dim_per_sample + dense_arch_layer_sizes[-1]

        crossnet = LowRankCrossNet(
            in_features=over_in_features,
            num_layers=dcn_num_layers,
            low_rank=dcn_low_rank_dim,
        )

        self.inter_arch = InteractionDCNArch(
            crossnet=crossnet,
        )

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits.
        """
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits


class DLRMTrain(nn.Module):
    """
    nn.Module to wrap DLRM model to use with train_pipeline.

    DLRM Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e, each EmbeddingBagConfig uses the same embedding_dim)

    Args:
        dlrm_module: DLRM module (DLRM or DLRM_Projection or DLRM_DCN) to be used in
        training

    Example::

        ebc = EmbeddingBagCollection(config=ebc_config)
        dlrm_module = DLRM(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )
        dlrm_model = DLRMTrain(dlrm_module)
    """

    def __init__(
        self,
        dlrm_module: DLRM_DCN,
    ) -> None:
        super().__init__()
        self.model = dlrm_module
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    def forward(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            batch: batch used with criteo and random data from torchrec.datasets
        Returns:
            Tuple[loss, Tuple[loss, logits, labels]]
        """
        logits = self.model(batch.dense_features, batch.sparse_features)
        logits = logits.squeeze(-1)
        loss = self.loss_fn(logits, batch.labels.float())

        return loss, (loss.detach(), logits.detach(), batch.labels.detach())

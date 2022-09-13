"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
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

from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import array_ops


def clip(embeddings, ids, max_norm):
    """Helper function for _embedding_lookup_and_transform.
    This function optionally clips embeddings to an l2-norm of max_norm.
    Args:
        embeddings: A `Tensor` of embeddings retrieved by `gather`.
        ids: The `ids` argument that was passed to `gather`.
        max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
            than this value.
    Returns:
        A `Tensor` with the same type as `embeddings`.
    """

    def _rank(x):
        """Helper function to retrieve the rank of a tensor.
        Args:
            x: Something convertible to `Tensor`.
        Returns:
            Either a pair `(rank, True)` where `rank` is an integer or a pair
            `(rank, False)` where `rank` is an integer `Tensor`. In either case,
            `rank` is the rank of `x`.
        """
        rank = ops.convert_to_tensor(x).get_shape().ndims
        if rank:
            return rank, True
        else:
            return array_ops.rank(x), False

    if max_norm is None:
        return embeddings
    ids_rank, ids_static = _rank(ids)
    embeddings_rank, embeddings_static = _rank(embeddings)
    return clip_ops.clip_by_norm(
        embeddings,
        max_norm,
        axes=(
            list(range(ids_rank, embeddings_rank))
            if ids_static and embeddings_static
            else math_ops.range(ids_rank, embeddings_rank)
        ),
    )

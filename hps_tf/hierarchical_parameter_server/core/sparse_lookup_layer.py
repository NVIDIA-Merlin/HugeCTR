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

import tensorflow as tf
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.nn import embedding_lookup

from hierarchical_parameter_server.core import lookup_ops


class SparseLookupLayer(tf.keras.layers.Layer):
    """
    Abbreviated as ``hps.SparseLookupLayer(*args, **kwargs)``.

    This is a wrapper class for HPS sparse lookup layer, which basically performs
    the same function as ``tf.nn.embedding_lookup_sparse``. Note that ``ps_config_file``
    and ``global_batch_size`` should be specified in the constructor if you want
    to use implicit HPS initialization.

    Parameters
    ----------
    model_name: str
            The name of the model that has embedding tables.
    table_id: int
            The index of the embedding table for the model specified by
            model_name.
    emb_vec_size: int
            The embedding vector size for the embedding table specified
            by model_name and table_id.
    emb_vec_dtype:
            The data type of embedding vectors which must be ``tf.float32``.
    ps_config_file: str
            The JSON configuration file for HPS initialization.
    global_batch_size: int
            The global batch size for HPS that is deployed on multiple GPUs.

    Examples
    --------
    .. code-block:: python

        import hierarchical_parameter_server as hps

        sparse_lookup_layer = hps.SparseLookupLayer(model_name = args.model_name,
                                                   table_id = args.table_id,
                                                   emb_vec_size = args.embed_vec_size,
                                                   emb_vec_dtype = tf.float32,
                                                   ps_config_file = args.ps_config_file,
                                                   global_batch_size = args.global_batch_size)

        @tf.function
        def _infer_step(inputs):
            embedding_vector = sparse_lookup_layer(sp_ids=inputs,
                                                  sp_weights = None,
                                                  combiner="mean")
            ...

        for i, (inputs, labels) in enumerate(dataset):
            _infer_step(inputs)
    """

    def __init__(
        self,
        model_name,
        table_id,
        emb_vec_size,
        emb_vec_dtype,
        ps_config_file="",
        global_batch_size=1,
        **kwargs,
    ):
        super(SparseLookupLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.table_id = table_id
        self.emb_vec_size = emb_vec_size
        self.emb_vec_dtype = emb_vec_dtype
        self.ps_config_file = ps_config_file
        self.global_batch_size = global_batch_size

    def call(self, sp_ids, sp_weights, name=None, combiner=None, max_norm=None):
        """
        Looks up embeddings for the given ids and weights from a list of tensors.
        This op assumes that there is at least one ID for each row in the dense tensor
        represented by ``sp_ids`` (i.e. there are no rows with empty features), and that
        all the indices of ``sp_ids`` are in canonical row-major order. The ``sp_ids``
        and ``sp_weights`` (if not None) are ``SparseTensor`` with rank of 2.
        Embeddings are always aggregated along the last dimension.
        If an ID value cannot be found in the HPS, the default embeddings are retrieved,
        which can be specified in the HPS configuration JSON file.

        Parameters
        ----------
        sp_ids:
            N x M ``SparseTensor`` of ``int32`` or ``int64`` IDs where N is typically batch size
            and M is arbitrary.
        sp_weights:
            Either a ``SparseTensor`` of float or double weights, or ``None`` to
            indicate all weights should be taken to be `1`. If specified, ``sp_weights``
            must have exactly the same shape and indices as ``sp_ids``.
        combiner:
            A string that specifies the reduction op:

            ``"sum"``
              Computes the weighted sum of the embedding results for each row.
            ``"mean"``
              Computes the weighted sum divided by the total weight.
            ``"sqrtn"``
              Computes the weighted sum divided by the square root of the sum of the
              squares of the weights.

            The default value is ``"mean"``.
        max_norm:
            if not ``None``, each embedding is clipped if its l2-norm is larger
            than this value, before combining.

        Returns
        -------
        emb_vector: ``tf.Tensor`` of float32
            A dense tensor representing the combined embeddings for the
            sparse IDs. For each row in the dense tensor represented by ``sp_ids``, the op
            looks up the embeddings for all IDs in that row, multiplies them by the
            corresponding weight, and combines these embeddings as specified.
            In other words, if

            .. code-block:: python

                shape(sp_ids) = shape(sp_weights) = [d0, d1]

            then

            .. code-block:: python

                shape(output) = [d0, self.emb_vec_dtype]

            For instance, if self.emb_vec_dtype is 16, and sp_ids / sp_weights are

            .. code-block:: python

                [0, 0]: id 1, weight 2.0
                [0, 1]: id 3, weight 0.5
                [1, 0]: id 0, weight 1.0
                [2, 3]: id 1, weight 3.0

            with ``combiner = "mean"``, then the output is a 3x16 matrix where

            .. code-block:: python

                output[0, :] = (vector_for_id_1 * 2.0 + vector_for_id_3 * 0.5) / (2.0 + 0.5)
                output[1, :] = (vector_for_id_0 * 1.0) / 1.0
                output[2, :] = (vector_for_id_1 * 3.0) / 3.0

        Raises
        ------
            TypeError: If ``sp_ids`` is not a ``SparseTensor``, or if ``sp_weights`` is
                neither ``None`` nor ``SparseTensor``.
            ValueError: If ``combiner`` is not one of {``"mean"``, ``"sqrtn"``, ``"sum"``}.

        """

        # Extract unique dense ids to be looked up
        if combiner is None:
            combiner = "mean"
        if combiner not in ("mean", "sqrtn", "sum"):
            raise ValueError(f"combiner must be one of 'mean', 'sqrtn' or 'sum', got {combiner}")

        if not isinstance(sp_ids, sparse_tensor.SparseTensor):
            raise TypeError(f"sp_ids must be SparseTensor, got {type(sp_ids)}")

        ignore_weights = sp_weights is None
        if not ignore_weights:
            if not isinstance(sp_weights, sparse_tensor.SparseTensor):
                raise TypeError(
                    f"sp_weights must be either None or SparseTensor," f"got {type(sp_weights)}"
                )
            sp_ids.values.get_shape().assert_is_compatible_with(sp_weights.values.get_shape())
            sp_ids.indices.get_shape().assert_is_compatible_with(sp_weights.indices.get_shape())
            sp_ids.dense_shape.get_shape().assert_is_compatible_with(
                sp_weights.dense_shape.get_shape()
            )
            # TODO(yleon): Add enhanced node assertions to verify that sp_ids and
            # sp_weights have equal indices and shapes.

        segment_ids = sp_ids.indices[:, 0]

        ids = sp_ids.values
        ids, idx = array_ops.unique(ids)

        # Query HPS for embeddings
        embeddings = lookup_ops.lookup(
            values=ids,
            model_name=self.model_name,
            table_id=self.table_id,
            emb_vec_size=self.emb_vec_size,
            emb_vec_dtype=self.emb_vec_dtype,
            ps_config_file=self.ps_config_file,
            global_batch_size=self.global_batch_size,
        )

        # Handle weights and combiner
        if not ignore_weights:
            if segment_ids.dtype != dtypes.int32:
                segment_ids = math_ops.cast(segment_ids, dtypes.int32)

            weights = sp_weights.values
            embeddings = array_ops.gather(embeddings, idx)

            original_dtype = embeddings.dtype
            if embeddings.dtype in (dtypes.float16, dtypes.bfloat16):
                # Cast low-precision embeddings to float32 during the computation to
                # avoid numerical issues.
                embeddings = math_ops.cast(embeddings, dtypes.float32)
            if weights.dtype != embeddings.dtype:
                weights = math_ops.cast(weights, embeddings.dtype)

            # Reshape weights to allow broadcast
            ones_shape = array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0)
            ones = array_ops.ones(ones_shape, dtype=dtypes.int32)
            bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones], 0)

            orig_weights_shape = weights.get_shape()
            weights = array_ops.reshape(weights, bcast_weights_shape)

            # Set the weight shape, since after reshaping to bcast_weights_shape,
            # the shape becomes None.
            if embeddings.get_shape().ndims is not None:
                weights.set_shape(
                    orig_weights_shape.concatenate(
                        [1 for _ in range(embeddings.get_shape().ndims - 1)]
                    )
                )

            embeddings *= weights

            if combiner == "sum":
                embeddings = math_ops.segment_sum(embeddings, segment_ids)
            elif combiner == "mean":
                embeddings = math_ops.segment_sum(embeddings, segment_ids)
                weight_sum = math_ops.segment_sum(weights, segment_ids)
                embeddings = math_ops.div_no_nan(embeddings, weight_sum)
            elif combiner == "sqrtn":
                embeddings = math_ops.segment_sum(embeddings, segment_ids)
                weights_squared = math_ops.pow(weights, 2)
                weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
                weight_sum_sqrt = math_ops.sqrt(weight_sum)
                embeddings = math_ops.div_no_nan(embeddings, weight_sum_sqrt)
            else:
                assert False, "Unrecognized combiner"
            if embeddings.dtype != original_dtype:
                embeddings = math_ops.cast(embeddings, original_dtype)
        else:
            if segment_ids.dtype not in (dtypes.int32, dtypes.int64):
                segment_ids = math_ops.cast(segment_ids, dtypes.int32)
            assert idx is not None
            if combiner == "sum":
                embeddings = math_ops.sparse_segment_sum(embeddings, idx, segment_ids)
            elif combiner == "mean":
                embeddings = math_ops.sparse_segment_mean(embeddings, idx, segment_ids)
            elif combiner == "sqrtn":
                embeddings = math_ops.sparse_segment_sqrt_n(embeddings, idx, segment_ids)
            else:
                assert False, "Unrecognized combiner"

        output_shape = [sp_ids.get_shape()[0], self.emb_vec_size]
        embeddings.set_shape(output_shape)
        return embeddings

"""
 Copyright (c) 2022, NVIDIA CORPORATION.
 
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

import numpy as np
import tensorflow as tf
import os
import math
from typing import Dict, List, Optional

from sparse_operation_kit import experiment as sok


class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_sizes, embedding_vec_size, num_gpus=1, use_tf=False, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self._vocab_sizes = vocab_sizes
        self._embedding_vec_size = embedding_vec_size
        self._use_tf = use_tf

        self._vars = []
        for size in vocab_sizes:
            if use_tf:
                shape = [size, self._embedding_vec_size]
                weights = tf.random.uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32)
                v = tf.Variable(weights)
            else:
                v = sok.DynamicVariable(dimension=self._embedding_vec_size)
            self._trainable_weights.append(v)
            self._vars.append(v)

    def call(self, inputs, training=True):
        fused_inputs = tf.split(inputs, num_or_size_splits=len(self._vocab_sizes), axis=1)
        emb_vectors = []
        for i in range(len(self._vars)):
            unique, indices = tf.unique(fused_inputs[i][:, 0])
            out = tf.nn.embedding_lookup(self._vars[i], unique)
            out = tf.gather(out, indices)
            emb_vectors.append(out)
        return emb_vectors


class MLP(tf.keras.layers.Layer):
    """Sequential multi-layer perceptron (MLP) block."""

    def __init__(
        self,
        units: List[int],
        use_bias: bool = True,
        activation="relu",
        final_activation=None,
        **kwargs,
    ):
        """Initializes the MLP layer.
        Args:
        units: Sequential list of layer sizes. List[int].
        use_bias: Whether to include a bias term.
        activation: Type of activation to use on all except the last layer.
        final_activation: Type of activation to use on last layer.
        **kwargs: Extra args passed to the Keras Layer base class.
        """
        super().__init__(**kwargs)

        self._sublayers = []
        for i, num_units in enumerate(units):
            kernel_init = tf.keras.initializers.GlorotNormal()
            bias_init = tf.keras.initializers.RandomNormal(stddev=math.sqrt(1.0 / num_units))

            self._sublayers.append(
                tf.keras.layers.Dense(
                    num_units,
                    use_bias=use_bias,
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    activation=activation if i < (len(units) - 1) else final_activation,
                )
            )

    def call(self, x: tf.Tensor):
        """Performs the forward computation of the block."""
        for layer in self._sublayers:
            x = layer(x)
        return x


class DotInteraction(tf.keras.layers.Layer):
    """Dot interaction layer.
    See theory in the DLRM paper: https://arxiv.org/pdf/1906.00091.pdf,
    section 2.1.3. Sparse activations and dense activations are combined.
    Dot interaction is applied to a batch of input Tensors [e1,...,e_k] of the
    same dimension and the output is a batch of Tensors with all distinct pairwise
    dot products of the form dot(e_i, e_j) for i <= j if self self_interaction is
    True, otherwise dot(e_i, e_j) i < j.
    Attributes:
        self_interaction: Boolean indicating if features should self-interact.
        If it is True, then the diagonal enteries of the interaction matric are
        also taken.
        skip_gather: An optimization flag. If it's set then the upper triangle part
        of the dot interaction matrix dot(e_i, e_j) is set to 0. The resulting
        activations will be of dimension [num_features * num_features] from which
        half will be zeros. Otherwise activations will be only lower triangle part
        of the interaction matrix. The later saves space but is much slower.
        name: String name of the layer.
    """

    def __init__(
        self, self_interaction: bool = False, skip_gather: bool = False, name=None, **kwargs
    ) -> None:
        self._self_interaction = self_interaction
        self._skip_gather = skip_gather
        super().__init__(name=name, **kwargs)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """Performs the interaction operation on the tensors in the list.
        The tensors represent as transformed dense features and embedded categorical
        features.
        Pre-condition: The tensors should all have the same shape.
        Args:
        inputs: List of features with shapes [batch_size, feature_dim].
        Returns:
        activations: Tensor representing interacted features. It has a dimension
        `num_features * num_features` if skip_gather is True, otherside
        `num_features * (num_features + 1) / 2` if self_interaction is True and
        `num_features * (num_features - 1) / 2` if self_interaction is False.
        """
        num_features = len(inputs)
        batch_size = tf.shape(inputs[0])[0]
        feature_dim = tf.shape(inputs[0])[1]
        # concat_features shape: batch_size, num_features, feature_dim
        try:
            concat_features = tf.concat(inputs, axis=-1)
            concat_features = tf.reshape(concat_features, [batch_size, -1, feature_dim])
        except (ValueError, tf.errors.InvalidArgumentError) as e:
            raise ValueError(
                f"Input tensors` dimensions must be equal, original" f"error message: {e}"
            )

        # Interact features, select lower-triangular portion, and re-shape.
        xactions = tf.matmul(concat_features, concat_features, transpose_b=True)
        ones = tf.ones_like(xactions, dtype=tf.float32)
        if self._self_interaction:
            # Selecting lower-triangular portion including the diagonal.
            lower_tri_mask = tf.linalg.band_part(ones, -1, 0)
            upper_tri_mask = ones - lower_tri_mask
            out_dim = num_features * (num_features + 1) // 2
        else:
            # Selecting lower-triangular portion not included the diagonal.
            upper_tri_mask = tf.linalg.band_part(ones, 0, -1)
            lower_tri_mask = ones - upper_tri_mask
            out_dim = num_features * (num_features - 1) // 2

        if self._skip_gather:
            # Setting upper tiangle part of the interaction matrix to zeros.
            activations = tf.where(
                condition=tf.cast(upper_tri_mask, tf.bool), x=tf.zeros_like(xactions), y=xactions
            )
            out_dim = num_features * num_features
        else:
            activations = tf.boolean_mask(xactions, lower_tri_mask)

        activations = tf.reshape(activations, (batch_size, out_dim))
        return activations


class DLRM(tf.keras.models.Model):
    def __init__(
        self,
        vocab_sizes: List[int],
        num_dense_features: int,
        embedding_vec_size: int,
        bottom_stack_units: List[int],
        top_stack_units: List[int],
        num_gpus=1,
        compress=False,
        use_tf=False,
        **kwargs,
    ):
        super(DLRM, self).__init__(**kwargs)
        self._vocab_sizes = vocab_sizes
        self._num_dense_features = num_dense_features
        self._embedding_vec_size = embedding_vec_size
        self._compress = compress
        self._use_tf = use_tf

        self._embedding_layer = Embedding(
            self._vocab_sizes, self._embedding_vec_size, num_gpus, use_tf=use_tf
        )
        self._bottom_stack = MLP(units=bottom_stack_units, final_activation="relu")
        self._feature_interaction = DotInteraction(self_interaction=False, skip_gather=False)
        self._top_stack = MLP(units=top_stack_units, final_activation=None)

    def call(self, inputs: List[tf.Tensor], training=True):
        dense_input = inputs[0]

        dense_embedding_vec = self._bottom_stack(dense_input)
        sparse_embeddings = self._embedding_layer(inputs[1], training=training)
        interaction_args = sparse_embeddings + [dense_embedding_vec]
        interaction_output = self._feature_interaction(interaction_args)
        feature_interaction_output = tf.concat([dense_embedding_vec, interaction_output], axis=1)

        prediction = self._top_stack(feature_interaction_output)
        return prediction
